# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Compartmentalised linear map: abstraction for atlas-based dimension reduction.
"""
import jax
import jax.numpy as jnp
from collections import OrderedDict
from functools import reduce
from typing import Callable, Literal, Mapping, Optional, Tuple
from ..engine import Tensor, standard_axis_number
from ..engine.paramutil import _to_jax_array


def normalise_mean(data: Tensor, weight: Tensor) -> Tensor:
    norm_fact = weight.sum(-1, keepdims=True)
    return data / norm_fact


def normalise_absmean(data: Tensor, weight: Tensor) -> Tensor:
    norm_fact = jnp.abs(weight).sum(-1, keepdims=True)
    return data / norm_fact


def normalise_zscore(data: Tensor, weight: Optional[Tensor] = None) -> Tensor:
    mean, std = data.mean(-1, keepdims=True), data.std(-1, keepdims=True)
    return (data - mean) / std


def normalise_psc(data: Tensor, weight: Optional[Tensor] = None) -> Tensor:
    mean = data.mean(-1, keepdims=True)
    return 100 * (data - mean) / mean


def _norm() -> Mapping[str, Callable]:
    return {
        'mean': normalise_mean,
        'absmean': normalise_absmean,
        'zscore': normalise_zscore,
        'psc': normalise_psc
    }


def form_dynamic_slice(
    shape: Tensor,
    slice_axis: int,
    slice_index: int,
    slice_size: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    ndim = len(shape)
    slice_axis = standard_axis_number(slice_axis, ndim)
    indices = tuple(0 if i != slice_axis else slice_index
                    for i in range(ndim))
    sizes = tuple(s if i != slice_axis else slice_size
                  for i, s in enumerate(shape))
    return indices, sizes


def select_compartment(
    input: Tensor,
    limit: Tuple[int, int],
) -> Tensor:
    shape = input.shape
    index, size = limit
    indices, sizes = form_dynamic_slice(shape, -2, index, size)
    return jax.lax.dynamic_slice(input, indices, sizes)


def linear_call(
    input: Tensor,
    weight: Tensor,
    forward_mode: Literal['map', 'project'] = 'map',
) -> Tensor:
    if forward_mode == 'map':
        return weight @ input
    elif forward_mode == 'project':
        return jnp.linalg.lstsq(weight.swapaxes(-2, -1), input)[0]


def _compartmentalised_linear_impl(
    input: Tensor,
    weight: Tensor,
    limits: Tuple[int, int],
    normalisation: Optional[str] = None,
    forward_mode: Literal['map', 'project'] = 'map',
) -> Tensor:
    """
    Linear layer with compartment-specific weights.
    """
    weight = _to_jax_array(weight)
    if weight.shape == (0,):
        return None
    compartment = select_compartment(input, limits)
    out = linear_call(compartment, weight, forward_mode)
    return _norm()[normalisation](out, weight) if normalisation else out


def concatenate_and_decode(
    data: Mapping[str, Tensor],
    decoder: Optional[Mapping[str, Tensor]] = None,
    shape: Optional[Tuple[int, ...]] = None,
    concatenate: bool = True,
) -> Tensor:
    if decoder is not None:
        out = jnp.empty(shape)
        for compartment, tensor in data.items():
            loc = (decoder[compartment] - 1)
            out = out.at[..., loc, :].set(tensor)
    elif concatenate:
        out = jnp.concatenate(tuple(v for v in data.values()), -2)
    return out


def compartmentalised_linear(
    input: Tensor,
    weight: Mapping[str, Tensor],
    limits: Mapping[str, Tuple[int, int]],
    decoder: Optional[Mapping[str, Tensor]] = None,
    bias: Optional[Tensor] = None,
    normalisation: (
        Optional[Literal['mean', 'absmean', 'zscore', 'psc']]) = None,
    forward_mode: Literal['map', 'project'] = 'map',
    concatenate: bool = True,
) -> Tensor:
    """
    Linear mapping with compartment-specific weights.

    Parameters
    ----------
    input : Tensor
        Input data.
    weight : Mapping[str, Tensor]
        Compartment-specific weights.
    limits : Mapping[str, Tuple[int, int]]
        Indices denoting the start and size of each compartment.
    decoder : Optional[Mapping[str, Tensor]] (default: None)
        Decoder for compartment-specific weights.
    bias : Optional[Tensor] (default: None)
        Bias term.
    normalisation : Optional[Literal['mean', 'absmean', 'zscore', 'psc']] (default: None)
        Normalisation approach for output.
    forward_mode : Literal['map', 'project'] (default: 'map')
        Forward mode for linear mapping.

    Returns
    -------
    Tensor
        Linearly mapped data.
    """
    out = OrderedDict()
    for k, v in weight.items():
        out[k] = _compartmentalised_linear_impl(
            input=input,
            weight=v,
            limits=limits[k],
            normalisation=normalisation,
            forward_mode=forward_mode,
        )
    out_shape = (
        *input.shape[:-2],
        reduce(
            lambda x, y: x + y,
            (v.shape[-2] for v in weight.values())
        ),
        input.shape[-1])
    out = concatenate_and_decode(
        data=out,
        decoder=decoder,
        shape=out_shape,
        concatenate=concatenate,
    )
    if bias is not None:
        out = out + bias[..., None]
    return out
