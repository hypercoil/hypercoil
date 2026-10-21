# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Compartmentalised linear map: abstraction for atlas-based dimension reduction.
"""
from __future__ import annotations
from collections import OrderedDict
from functools import reduce
from typing import (
    Callable,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp

from ..engine import Tensor, standard_axis_number
from ..engine.paramutil import _to_jax_array


def normalise_mean(data: Tensor, weight: Tensor) -> Tensor:
    """Divide ``data`` by the row sums of ``weight``."""
    norm_fact = weight.sum(-1, keepdims=True)
    return data / norm_fact


def normalise_absmean(data: Tensor, weight: Tensor) -> Tensor:
    """Divide ``data`` by the row sums of the absolute ``weight``."""
    norm_fact = jnp.abs(weight).sum(-1, keepdims=True)
    return data / norm_fact


def normalise_zscore(data: Tensor, weight: Optional[Tensor] = None) -> Tensor:
    """Z-score ``data`` along the last axis; ``weight`` is unused."""
    mean, std = data.mean(-1, keepdims=True), data.std(-1, keepdims=True)
    return (data - mean) / std


def normalise_psc(data: Tensor, weight: Optional[Tensor] = None) -> Tensor:
    """Convert ``data`` to percent change from its mean; ``weight`` unused."""
    mean = data.mean(-1, keepdims=True)
    return 100 * (data - mean) / mean


def _norm() -> Mapping[str, Callable]:
    return {
        'mean': normalise_mean,
        'absmean': normalise_absmean,
        'zscore': normalise_zscore,
        'psc': normalise_psc,
    }


def form_dynamic_slice(
    shape: Sequence[int],
    slice_axis: int,
    slice_index: int,
    slice_size: int,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Form the start indices and sizes of a slice along one axis.

    Parameters
    ----------
    shape
        Shape of the tensor to slice.
    slice_axis
        Axis along which to slice.
    slice_index
        Start index of the slice along ``slice_axis``.
    slice_size
        Size of the slice along ``slice_axis``.

    Returns
    -------
    tuple
        Start indices and slice sizes for :func:`jax.lax.dynamic_slice`: 0 and
        the full size on every axis but ``slice_axis``.
    """
    ndim = len(shape)
    slice_axis = standard_axis_number(slice_axis, ndim)
    indices = tuple(0 if i != slice_axis else slice_index for i in range(ndim))
    sizes = tuple(
        s if i != slice_axis else slice_size for i, s in enumerate(shape)
    )
    return indices, sizes


def select_compartment(
    input: Tensor,
    limit: Tuple[int, int],
) -> Tensor:
    """Slice ``limit = (start, size)`` from axis -2 of ``input``."""
    shape = input.shape
    index, size = limit
    indices, sizes = form_dynamic_slice(shape, -2, index, size)
    return jax.lax.dynamic_slice(input, indices, sizes)


def linear_call(
    input: Tensor,
    weight: Tensor,
    forward_mode: Literal['map', 'project'] = 'map',
) -> Tensor:
    """
    Apply a linear map to an input, or project the input onto the map.

    Parameters
    ----------
    input
        Input of shape ``(*, V, obs)``.
    weight
        Linear map of shape ``(L, V)``.
    forward_mode
        ``'map'`` returns ``weight @ input``; ``'project'`` returns the
        least-squares solution of ``weight.T @ out = input``.

    Returns
    -------
    Tensor
        Output of shape ``(*, L, obs)``; ``None`` for any other
        ``forward_mode``.
    """
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
    """
    Concatenate per-compartment outputs, optionally in decoder order.

    Parameters
    ----------
    data
        Output of each compartment, keyed by compartment name, with labels on
        axis -2.
    decoder
        For each compartment, the 1-based positions of its labels in the
        output. If ``None``, the compartments are concatenated in order.
    shape
        Shape of the output; required with ``decoder``.
    concatenate
        If False, ``data`` is returned unchanged.

    Returns
    -------
    Tensor or Mapping
        The concatenated output, or ``data`` if ``concatenate`` is False.
    """
    if not concatenate:
        return data
    if decoder is not None:
        out = jnp.empty(shape)
        for compartment, tensor in data.items():
            loc = decoder[compartment] - 1
            out = out.at[..., loc, :].set(tensor)
    else:
        out = jnp.concatenate(tuple(v for v in data.values()), -2)
    return out


def encode_and_bias(
    basis: Union[Tensor, Mapping[str, Tensor]],
    input: Tensor,
    limits: Optional[Mapping[str, Tuple[int, int]]] = None,
    weight: Optional[Mapping[str, Tensor]] = None,
    bias: Optional[Tensor] = None,
    encode: bool = False,
) -> Tensor:
    """
    Optionally correlate a basis with an input, then add a bias.

    Parameters
    ----------
    basis
        Tensor, or mapping of tensors keyed by compartment.
    input
        Input whose compartments are correlated with ``basis`` if ``encode``.
    limits
        ``(start, size)`` of each compartment along axis -2 of ``input``.
    weight
        Mapping keyed by compartment; its keys select the compartments when
        ``basis`` is a single tensor and ``limits`` is given.
    bias
        Bias added along the last axis, or a mapping of biases keyed by
        compartment.
    encode
        Replace ``basis`` by its correlation (:func:`pairedcorr`) with each
        compartment of ``input``.

    Returns
    -------
    Tensor or Mapping
        The encoded and biased basis, in the form of ``basis``.
    """
    out = basis
    if encode:
        from .cov import pairedcorr
        if isinstance(basis, Mapping):
            out = {k: pairedcorr(
                select_compartment(input, limits[k]), v
            ) for k, v in out.items()}
        elif limits is not None:
            out = {k: pairedcorr(
                select_compartment(input, limits[k]), out
            ) for k in weight}
        else:
            out = pairedcorr(input, out)
    if bias is not None:
        if isinstance(out, Mapping):
            out = {k: out[k] + bias[k][..., None] for k in out}
        else:
            out = out + bias[..., None]
    return out


def compartmentalised_linear(
    input: Tensor,
    weight: Mapping[str, Tensor],
    limits: Mapping[str, Tuple[int, int]],
    decoder: Optional[Mapping[str, Tensor]] = None,
    bias: Optional[Tensor] = None,
    normalisation: (
        Optional[Literal['mean', 'absmean', 'zscore', 'psc']]
    ) = None,
    forward_mode: Literal['map', 'project'] = 'map',
    concatenate: bool = True,
    encode: bool = False,
) -> Tensor:
    """
    Linear mapping with compartment-specific weights.

    Parameters
    ----------
    input
        Input data.
    weight
        Compartment-specific weights.
    limits
        Indices denoting the start and size of each compartment.
    decoder
        Decoder for compartment-specific weights.
    bias
        Bias term.
    normalisation
        Normalisation approach for output.
    forward_mode
        Forward mode for linear mapping.
    concatenate
        Concatenate the compartment outputs into one tensor; otherwise return
        a mapping by compartment.
    encode
        Encode ``input`` by its Pearson correlation with each mapped
        time series.

    Returns
    -------
    Tensor
        Linearly mapped data.
    """
    out = OrderedDict()
    for k, v in weight.items():
        if v.shape == (0,):
            continue
        out[k] = _compartmentalised_linear_impl(
            input=input,
            weight=v,
            limits=limits[k],
            normalisation=normalisation,
            forward_mode=forward_mode,
        )
    out_shape = (
        *input.shape[:-2],
        reduce(lambda x, y: x + y, (v.shape[-2] for v in out.values())),
        input.shape[-1],
    )
    out = concatenate_and_decode(
        data=out,
        decoder=decoder,
        shape=out_shape,
        concatenate=concatenate,
    )
    return encode_and_bias(
        basis=out,
        input=input,
        limits=limits,
        weight=weight,
        bias=bias,
        encode=encode,
    )
