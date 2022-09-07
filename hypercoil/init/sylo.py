# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Initialisations for sylo-based neural networks.
"""
import math
import jax
import jax.numpy as jnp
import distrax
from typing import Literal, Optional, Tuple, Type, Union
from .base import MappedInitialiser, retrieve_parameter
from .mapparam import MappedParameter
from ..functional.utils import PyTree, Tensor


def calculate_gain(
    nonlinearity: str,
    negative_slope: Optional[float] = None
) -> float:
    """
    Port from PyTorch.

    See ``torch.nn.init.calculate_gain``.
    """
    nonlinearity = nonlinearity.lower()
    gain_map = {
        'linear': 1.,
        'conv1d': 1.,
        'conv2d': 1.,
        'conv3d': 1.,
        'conv_transpose1d': 1.,
        'conv_transpose2d': 1.,
        'conv_transpose3d': 1.,
        'tanh': 5.0 / 3,
        'selu': 3.0 / 4,
        'leaky_relu': (2.0 / (1 + negative_slope ** 2))
    }
    gain = gain_map.get(nonlinearity, None)
    if gain is None:
        raise ValueError(f'Unsupported nonlinearity {nonlinearity}')
    return gain


#TODO: mark this as experimental.
#TODO: This needs a lot of review/revision with a proper derivation.
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

    if shape_R is None: shape_R = shape
    gain = calculate_gain(
        nonlinearity=nonlinearity,
        negative_slope=negative_slope
    )
    fan_crosshair = _calculate_correct_fan_crosshair(
        shape=shape, shape_R=shape_R, mode=mode)
    fan_expansion = _calculate_fan_in_expansion(
        shape=shape,
        psd=psd
    )

    # TODO: Does gain go inside or outside of the outer sqrt?
    # Right now it's outside since we'd rather the std explode than vanish...
    std = gain / math.sqrt(math.sqrt(fan_crosshair * fan_expansion))
    if init_distr == 'normal':
        distr = distrax.Normal(loc=0, scale=std)
    elif init_distr == 'uniform':
        bound = math.sqrt(3.) * std
        distr = distrax.Uniform(low=-bound, high=bound)

    if psd:
        return distr.sample(seed=key, sample_shape=shape)
    else:
        key_L, key_R = jax.random.split(key)
        return (
            distr.sample(seed=key_L, sample_shape=shape),
            distr.sample(seed=key_R, sample_shape=shape_R)
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
    shape_R: Tuple[int, ...]
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
        return rank + (rank ** 2) / matrix_dim
    return rank


class SyloInitialiser(MappedInitialiser):
    negative_slope: float = 0
    mode: Literal['fan_in', 'fan_out'] = 'fan_in'
    init_distr: Literal['uniform', 'normal'] = 'uniform'
    nonlinearity: str = 'leaky_relu'
    psd: bool = False

    def __init__(
        self,
        negative_slope:float = 0,
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
        param_name: str = "weight",
        key: jax.random.PRNGKey,
        **params,
    ):
        params_init = ()
        parameters = retrieve_parameter(model, param_name=param_name)
        keys = jax.random.split(key, len(parameters))
        for key, parameter in zip(keys, parameters):
            if (not isinstance(parameter, jnp.DeviceArray) and
                not hasattr(parameter, '__jax_array__')):
                shape = (parameter[0].shape, parameter[1].shape)
            else:
                shape = parameter.shape
            params_init += (self._init(
                shape=shape,
                key=key,
                **params,
            ),)
        return params_init

    def _init(
        self,
        shape=Union[Tuple[int, ...], Tuple[Tuple[int, ...], Tuple[int, ...]]],
        key=jax.random.PRNGKey
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
        negative_slope:float = 0,
        mode: Literal['fan_in', 'fan_out'] = 'fan_in',
        init_distr: Literal['uniform', 'normal'] = 'uniform',
        nonlinearity: str = 'leaky_relu',
        psd: bool = False,
        param_name: str = 'weight',
        key: jax.random.PRNGKey,
        **params,
    ):
        init = cls(
            mapper=mapper,
            negative_slope=negative_slope,
            mode=mode,
            init_distr=init_distr,
            nonlinearity=nonlinearity,
            psd=psd,
        )
        return super()._init_impl(
            init=init, model=model, param_name=param_name, key=key, **params,
        )


def sylo_init_(*pparams, **params):
    """Kaiming-like initialisation for sylo module.

    Notes
    -----
    * Revisiting these notes some years later, I think this needs to be
      revisited before this functionality is migrated out of the experimental.
    * The overall fan is the square root of the product of
      (1) the fan from raw weights (H x r, W x r) to expanded template
          (H x W),
      (2) the fan from expanded template to the output feature map.
    * Fan (1) is the rank r of the raw weights. If the operation is symmetric
      (the left and right weights are identical), we need to add an additional
      term that scales quadratically with the rank and inversely with the raw
      weight dimension (H = W). It's possible that these are the first two
      terms of a longer series, but I'm not currently sure how to figure this
      out.
    * Fan (2) is computed as for convolutional networks: it's the product of
      the number of channels and the receptive field size, which here is the
      size of the crosshair (H + W - 1).
    * I hacked this out empirically (don't do this...), so there's a good
      chance that we can do better once we come up with a good theoretical
      justification.
    * I believe that we need to take a double square root because the template
      is a product of the raw weights.
    """
    raise NotImplementedError(
        'In-place initialisation is deprecated. Use ``sylo_init``.')
