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
from ..functional import complex_recompose, product_filtfilt
from ..init.freqfilter import FreqFilterInitialiser, FreqFilterSpec


class FrequencyDomainFilter(eqx.Module):
    r"""
    Filtering or convolution via transfer function multiplication in the
    frequency domain.

    Each time series in the input dataset is transformed into the frequency
    domain, where it is multiplied by the complex-valued transfer function of
    each filter in the module's bank. Each filtered frequency spectrum is then
    transformed back into the time domain. To ensure a zero-phase filter, the
    filtered time series are reversed and the process is repeated.

    :Dimension: **Input :** :math:`(N, *, C, T)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, :math:`C` denotes number of data
                    channels or variables, T denotes number of time points or
                    observations per channel.
                **Output :** :math:`(N, *, F, C, T)`
                    F denotes number of filters.

    Parameters
    ----------
    filter_specs : list(FreqFilterSpec)
        A list of filter specifications implemented as
        :doc:`FreqFilterSpec <hypercoil.init.FreqFilterSpec>`
        objects. These determine the filter bank that
        is applied to the input. Consult the ``FreqFilterSpec`` documentation
        for further details.
    dim : int or None
        Number of frequency bins. This must be conformant with the time series
        supplied as input. If you are uncertain about the dimension in the
        frequency domain, it is possible to instead provide the ``time_dim``
        argument (the length of the time series), but either ``time_dim`` or
        ``dim`` (but not both) must be specified.
    time_dim : int or None
        Number of time points in the input time series. Either ``time_dim`` or
        ``dim`` (but not both) must be specified.
    filter : callable (default ``product_filtfilt``)
        Callable function that takes as its arguments an input time series and
        a set of transfer functions. It transforms the input into the frequency
        domain, multiplies it by the transfer function bank, and transforms it
        back. By default, the ``product_filtfilt`` function is used to ensure a
        zero-phase filter.
    domain : Domain object (default ``AmplitudeAtanh``)
        A domain object from ``hypercoil.init.domain``, used to specify
        the domain of the filter spectrum. An
        :doc:`Identity <hypercoil.init.domainbase.Identity>` object yields the
        raw transfer function, while an
        :doc:`AmplitudeAtanh <hypercoil.init.domain.AmplitudeAtanh>` object
        transforms the amplitudes of each bin by the inverse tanh (atanh)
        function prior to convolution with the input. Using the
        ``AmplitudeAtanh`` domain thereby constrains transfer function
        amplitudes to [0, 1) and prevents explosive gain. An
        :doc:`AmplitudeMultiLogit <hypercoil.init.domain.AmplitudeMultiLogit>`
        domain can be used to instantiate and learn a parcellation over
        frequencies.

    Attributes
    ----------
    preweight : Tensor :math:`(F, D)`
        Filter bank transfer functions in the module's domain. F denotes the
        total number of filters in the bank, and D denotes the dimension of
        the input dataset in the frequency domain. The weights are initialised
        to emulate each  of the filters specified in the ``filter_specs``
        parameter following the ``freqfilter_init_`` function.
    weight : Tensor :math:`(F, D)`
        The transfer function weights as seen by the input dataset in the
        frequency domain. This entails mapping the weights out of the
        specified predomain and applying any clamps declared in the input
        specifications.
    clamp_points : Tensor :math:`(F, D)`
        Boolean-valued tensor mask indexing points in the transfer function
        that should be clamped to particular values. Any points so indexed
        will not be learnable. If this is None, then no clamp is applied.
    clamp_values : Tensor :math:`(V)`
        Tensor containing values to which the transfer functions are clamped.
        `V` denotes the total number of values to be clamped across all
        transfer functions. If this is None, then no clamp is applied.
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
        key: "jax.random.PRNGKey",
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
        key: "jax.random.PRNGKey",
    ) -> "FrequencyDomainFilter":

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
                clamp_name="clamp",
                key=jax.random.PRNGKey(0),
            )
        return model

    @staticmethod
    def _set_dimension(freq_dim: int, time_dim: int) -> int:
        if freq_dim is None:
            if time_dim is None:
                raise ValueError(
                    "You must specify the dimension in either "
                    "the frequency or time domain"
                )
            else:
                dim = time_dim // 2 + 1
        else:
            dim = freq_dim
        return dim

    @property
    def clamped_weight(self) -> Tensor:
        weight = _to_jax_array(self.weight)
        if self.clamp is not None:
            clamp_points, clamp_values = self.clamp
            weight = jnp.where(clamp_points, clamp_values, weight)
        return weight

    def __call__(
        self,
        input: Tensor,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
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
