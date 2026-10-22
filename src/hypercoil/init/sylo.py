# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Initialisations for sylo-based neural networks.
"""
from __future__ import annotations
import math
from typing import Callable, Literal, Optional, Tuple, Type, Union

import jax
from numpyro.distributions import Normal, Uniform

from ..engine import PyTree, Tensor
from .base import MappedInitialiser, retrieve_address
from .mapparam import MappedParameter


def calculate_gain(
    nonlinearity: str,
    negative_slope: Optional[float] = None,
) -> float:
    """
    Return the recommended gain for a nonlinearity, as in PyTorch.

    The values follow ``torch.nn.init.calculate_gain``.

    Parameters
    ----------
    nonlinearity
        One of ``'linear'``, ``'conv1d'``, ``'conv2d'``, ``'conv3d'``,
        ``'conv_transpose1d'``, ``'conv_transpose2d'``,
        ``'conv_transpose3d'``, ``'tanh'``, ``'selu'`` and ``'leaky_relu'``;
        case is ignored.
    negative_slope
        Negative slope of ``'leaky_relu'``; required for ``'leaky_relu'``.

    Returns
    -------
    float
        Gain.

    Raises
    ------
    ValueError
        If ``nonlinearity`` is not supported.
    """
    nonlinearity = nonlinearity.lower()
    gain_map = {
        'linear': 1.0,
        'conv1d': 1.0,
        'conv2d': 1.0,
        'conv3d': 1.0,
        'conv_transpose1d': 1.0,
        'conv_transpose2d': 1.0,
        'conv_transpose3d': 1.0,
        'tanh': 5.0 / 3,
        'selu': 3.0 / 4,
        'leaky_relu': (2.0 / (1 + negative_slope**2)),
    }
    gain = gain_map.get(nonlinearity, None)
    if gain is None:
        raise ValueError(f'Unsupported nonlinearity {nonlinearity}')
    return gain


# TODO: mark this as experimental.
# TODO: This needs a lot of review/revision with a proper derivation.
def sylo_init(
    *,
    shape: Tuple[int, ...],
    shape_R: Optional[Tuple[int, ...]] = None,
    negative_slope: float = 0,
    mode: Literal['fan_in', 'fan_out'] = 'fan_in',
    init_distr: Literal['uniform', 'normal'] = 'uniform',
    nonlinearity: str = 'leaky_relu',
    psd: bool = False,
    key: jax.random.PRNGKey,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:

    r"""
    Sample the generators of a sylo layer with a fan-scaled variance.

    The standard deviation is
    :math:`g / \sqrt{\sqrt{f_c f_e}}` for gain :math:`g`
    (:func:`calculate_gain`). The crosshair fan :math:`f_c` is
    :math:`C_{in} (H + H_R - 1)` for ``'fan_in'`` and
    :math:`C_{out} (H + H_R - 1)` for ``'fan_out'``. The expansion fan
    :math:`f_e` is the rank :math:`r`, or :math:`r + r^2 / H` if ``psd``.

    Parameters
    ----------
    shape
        Shape ``(C_out, C_in, H, r)`` of the left generator.
    shape_R
        Shape of the right generator. Defaults to ``shape``.
    negative_slope
        Negative slope of a leaky ReLU nonlinearity.
    mode
        ``'fan_in'`` or ``'fan_out'``: the fan that scales the variance.
    init_distr
        ``'uniform'`` (bounds :math:`\pm \sqrt{3}` times the standard
        deviation) or ``'normal'``.
    nonlinearity
        Nonlinearity whose gain scales the standard deviation.
    psd
        Sample one generator, for a symmetric positive semidefinite template.
    key
        Pseudo-random number generator key.

    Returns
    -------
    Tensor or tuple
        The generator of shape ``shape`` if ``psd``; otherwise the left and
        right generators.
    """
    if shape_R is None:
        shape_R = shape
    gain = calculate_gain(
        nonlinearity=nonlinearity,
        negative_slope=negative_slope,
    )
    fan_crosshair = _calculate_correct_fan_crosshair(
        shape=shape, shape_R=shape_R, mode=mode
    )
    fan_expansion = _calculate_fan_in_expansion(
        shape=shape,
        psd=psd,
    )

    # TODO: Does gain go inside or outside of the outer sqrt?
    # Right now it's outside since we'd rather the std explode than vanish...
    std = gain / math.sqrt(math.sqrt(fan_crosshair * fan_expansion))
    if init_distr == 'normal':
        distr = Normal(loc=0, scale=std)
    elif init_distr == 'uniform':
        bound = math.sqrt(3.0) * std
        distr = Uniform(low=-bound, high=bound)

    if psd:
        return distr.sample(key=key, sample_shape=shape)
    else:
        key_L, key_R = jax.random.split(key)
        return (
            distr.sample(key=key_L, sample_shape=shape),
            distr.sample(key=key_R, sample_shape=shape_R),
        )


def _calculate_correct_fan_crosshair(
    shape: Tuple[int, ...],
    shape_R: Tuple[int, ...],
    mode: Literal['fan_in', 'fan_out'],
) -> int:
    mode = mode.lower()
    fan_in, fan_out = _calculate_fan_in_and_fan_out_crosshair(shape, shape_R)
    return fan_in if mode == 'fan_in' else fan_out


def _calculate_fan_in_and_fan_out_crosshair(
    shape: Tuple[int, ...],
    shape_R: Tuple[int, ...],
) -> Tuple[int, int]:
    num_output_fmaps, num_input_fmaps = shape[:2]
    receptive_field_size = shape[-2] + shape_R[-2] - 1
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_fan_in_expansion(shape: Tuple[int, ...], psd: bool) -> float:
    rank = shape[-1]
    # It could also have a value like ``skew`` or ``cross`` if we decide
    # to pass some symmetry specification -- thus this weird-looking
    # expression
    if psd is True:
        matrix_dim = shape[-2]
        return rank + (rank**2) / matrix_dim
    return rank


class SyloInitialiser(MappedInitialiser):
    """
    Initialiser for the generators of a sylo layer.

    See :func:`sylo_init` and :class:`MappedInitialiser` for argument details.

    Parameters
    ----------
    negative_slope
        Negative slope of a leaky ReLU nonlinearity.
    mode
        ``'fan_in'`` or ``'fan_out'``: the fan that scales the variance.
    init_distr
        ``'uniform'`` or ``'normal'``.
    nonlinearity
        Nonlinearity whose gain scales the standard deviation.
    psd
        Sample one generator, for a symmetric positive semidefinite template.
    mapper
        Parameter map applied after initialisation.
    """
    negative_slope: float = 0
    mode: Literal['fan_in', 'fan_out'] = 'fan_in'
    init_distr: Literal['uniform', 'normal'] = 'uniform'
    nonlinearity: str = 'leaky_relu'
    psd: bool = False

    def __init__(
        self,
        negative_slope: float = 0,
        mode: Literal['fan_in', 'fan_out'] = 'fan_in',
        init_distr: Literal['uniform', 'normal'] = 'uniform',
        nonlinearity: str = 'leaky_relu',
        psd: bool = False,
        mapper: Optional[Type[MappedParameter]] = None,
    ):
        self.negative_slope = negative_slope
        self.mode = mode
        self.init_distr = init_distr
        self.nonlinearity = nonlinearity
        self.psd = psd
        super().__init__(mapper=mapper)

    def __call__(
        self,
        model: PyTree,
        *,
        where: Union[str, Callable] = 'weight',
        key: jax.random.PRNGKey,
        **params,
    ):
        params_init = ()
        parameters = retrieve_address(model, where=where)
        keys = jax.random.split(key, len(parameters))
        for key, parameter in zip(keys, parameters):
            if not isinstance(parameter, jax.Array) and not hasattr(
                parameter, '__jax_array__'
            ):
                shape = (parameter[0].shape, parameter[1].shape)
            else:
                shape = parameter.shape
            params_init += (
                self._init(
                    shape=shape,
                    key=key,
                    **params,
                ),
            )
        return params_init

    def _init(
        self,
        shape=Union[Tuple[int, ...], Tuple[Tuple[int, ...], Tuple[int, ...]]],
        key=jax.random.PRNGKey,
    ):
        if isinstance(shape[0], tuple):
            shape, shape_R = shape
        else:
            shape_R = None
        return sylo_init(
            shape=shape,
            shape_R=shape_R,
            negative_slope=self.negative_slope,
            mode=self.mode,
            init_distr=self.init_distr,
            nonlinearity=self.nonlinearity,
            psd=self.psd,
            key=key,
        )

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Optional[Type[MappedParameter]] = None,
        negative_slope: float = 0,
        mode: Literal['fan_in', 'fan_out'] = 'fan_in',
        init_distr: Literal['uniform', 'normal'] = 'uniform',
        nonlinearity: str = 'leaky_relu',
        psd: bool = False,
        where: Union[str, Callable] = 'weight',
        key: jax.random.PRNGKey,
        **params,
    ):
        """Initialise the parameters at ``where`` by :func:`sylo_init`."""
        init = cls(
            mapper=mapper,
            negative_slope=negative_slope,
            mode=mode,
            init_distr=init_distr,
            nonlinearity=nonlinearity,
            psd=psd,
        )
        return super()._init_impl(
            init=init,
            model=model,
            where=where,
            key=key,
            **params,
        )
