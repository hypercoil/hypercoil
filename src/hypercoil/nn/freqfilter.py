# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Modules supporting filtering/convolution as a product in the frequency domain.
"""
from __future__ import annotations
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

from ..engine import Tensor
from ..engine.paramutil import _to_jax_array
from ..functional import product_filtfilt
from ..init.freqfilter import FreqFilterInitialiser, FreqFilterSpec


class FrequencyDomainFilter(eqx.Module):
    r"""
    Filter bank applied by multiplication in the frequency domain.

    Each input time series is transformed into the frequency domain,
    multiplied by the transfer function of each filter in the bank, and
    transformed back. With the default ``filter``, :func:`product_filtfilt`,
    the filtered series is reversed and filtered again, giving a zero-phase
    filter. :meth:`from_specs` builds a bank from filter specifications.

    :Dimension: **Input :** :math:`(N, *, C, T)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, :math:`C` denotes number of data
                    channels or variables, T denotes number of time points or
                    observations per channel.
                **Output :** :math:`(N, *, F, C, T)`
                    F denotes number of filters.

    Parameters
    ----------
    num_channels
        Number of filters in the bank, :math:`F`.
    clamp_points
        Boolean mask of shape :math:`(F, D)` marking transfer function bins
        held at ``clamp_values``; clamped bins are not learnable. The clamp
        is applied only if both ``clamp_points`` and ``clamp_values`` are
        given.
    clamp_values
        Values of the clamped bins, broadcastable to :math:`(F, D)`.
    freq_dim
        Number of frequency bins, :math:`D`. Takes precedence over
        ``time_dim``.
    time_dim
        Number of time points :math:`T` in the input; gives
        :math:`D = \lfloor T / 2 \rfloor + 1`. One of ``freq_dim`` and
        ``time_dim`` is required.
    filter
        Function of the input time series and the transfer function bank
        that returns the filtered series.
    key
        Pseudo-random number generator key. Transfer function amplitudes are
        initialised as :math:`0.5 + 0.01 \epsilon` with standard normal
        :math:`\epsilon`.

    Raises
    ------
    ValueError
        If neither ``freq_dim`` nor ``time_dim`` is given.
    """
    weight: Tensor
    clamp: Optional[Tuple[Tensor, Tensor]]
    filter: Callable = product_filtfilt
    num_channels: int
    dim: int

    def __init__(
        self,
        num_channels: int,
        clamp_points: Optional[Tensor] = None,
        clamp_values: Optional[Tensor] = None,
        freq_dim: Optional[int] = None,
        time_dim: Optional[int] = None,
        filter: Callable = product_filtfilt,
        *,
        key: 'jax.random.PRNGKey',
    ):
        self.dim = self._set_dimension(freq_dim, time_dim)
        self.num_channels = num_channels
        self.filter = filter

        akey, pkey = jax.random.split(key, 2)
        amplitude = (
            0.01 * jax.random.normal(akey, (num_channels, self.dim)) + 0.5
        )
        # phase = 0.01 * jax.random.normal(
        #     pkey, (num_channels, self.dim)) + 0.5
        self.weight = amplitude  # complex_recompose(amplitude, phase)

        if clamp_points is not None and clamp_values is not None:
            self.clamp = (clamp_points, clamp_values)
        else:
            self.clamp = None

    @classmethod
    def from_specs(
        cls,
        filter_specs: List[FreqFilterSpec],
        freq_dim: Optional[int] = None,
        time_dim: Optional[int] = None,
        filter: Callable = product_filtfilt,
        *,
        key: 'jax.random.PRNGKey',
    ) -> 'FrequencyDomainFilter':

        """
        Create a filter initialised from filter specifications.

        Parameters
        ----------
        filter_specs
            Filter specifications; each critical frequency in each
            specification gives one channel. Clamps in any specification are
            applied.
        freq_dim
            Number of frequency bins.
        time_dim
            Number of time points; gives ``time_dim // 2 + 1`` frequency bins
            if ``freq_dim`` is None.
        filter
            Filter function applied in the frequency domain.
        key
            Pseudo-random number generator key.

        Returns
        -------
        FrequencyDomainFilter
            Initialised filter.

        Raises
        ------
        ValueError
            If both ``freq_dim`` and ``time_dim`` are None.
        """
        num_channels = sum([len(s.Wn) for s in filter_specs])
        clamp = any([s.clamps is not None for s in filter_specs])
        freq_dim = cls._set_dimension(freq_dim, time_dim)
        if clamp:
            clamp_points, clamp_values = (
                jnp.empty(
                    (num_channels, freq_dim),
                    dtype=jnp.complex64,
                ),
                jnp.empty(
                    (num_channels, freq_dim),
                    dtype=jnp.complex64,
                ),
            )
        else:
            clamp_points, clamp_values = None, None

        init_key, null_key = jax.random.split(key)
        model = cls(
            num_channels=num_channels,
            freq_dim=freq_dim,
            time_dim=None,
            filter=filter,
            clamp_points=clamp_points,
            clamp_values=clamp_values,
            key=null_key,
        )

        wkey, ckey = jax.random.split(init_key, 2)
        model = FreqFilterInitialiser.init(
            model,
            filter_specs=filter_specs,
            key=wkey,
        )
        if clamp:
            model = FreqFilterInitialiser.init(
                model,
                filter_specs=filter_specs,
                clamp_name='clamp',
                key=jax.random.PRNGKey(0),
            )
        return model

    @staticmethod
    def _set_dimension(freq_dim: int, time_dim: int) -> int:
        if freq_dim is None:
            if time_dim is None:
                raise ValueError(
                    'You must specify the dimension in either '
                    'the frequency or time domain'
                )
            else:
                dim = time_dim // 2 + 1
        else:
            dim = freq_dim
        return dim

    @property
    def clamped_weight(self) -> Tensor:
        """Weight with the clamped bins set to their clamped values."""
        weight = _to_jax_array(self.weight)
        if self.clamp is not None:
            clamp_points, clamp_values = self.clamp
            weight = jnp.where(clamp_points, clamp_values, weight)
        return weight

    def __call__(
        self,
        input: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        """
        Transform the input into the frequency domain, filter it, and
        transform the filtered signal back.
        """
        if input.ndim > 1 and input.shape[-2] > 1:
            input = input[..., None, :, :]
            weight = self.clamped_weight[..., None, :]
        else:
            weight = self.clamped_weight
        return self.filter(input, weight)
