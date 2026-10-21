# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility functions.
"""
from __future__ import annotations
from typing import Callable, Sequence, Tuple, Union

import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from ..engine.paramutil import Tensor


# TODO: This will not work if JAX ever adds sparse formats other than BCOO.
def is_sparse(X):
    return isinstance(X, BCOO)


def _conform_vector_weight(weight: Tensor) -> Tensor:
    if weight.ndim == 1:
        return weight
    if weight.shape[-2] != 1:
        return weight[..., None, :]
    return weight


def conform_mask(
    tensor: Tensor,
    mask: Tensor,
    axis: Sequence[int],
    batch=False,
) -> Tensor:
    """
    Conform a mask or weight for elementwise applying to a tensor.

    Parameters
    ----------
    tensor
        Tensor to which the mask is applied.
    mask
        Mask or weight, aligned with ``axis`` of ``tensor``.
    axis
        Axes of ``tensor`` spanned by ``mask``. For an int, ``mask`` is tiled
        over the leading axes.
    batch
        The first axis of ``mask`` is a batch axis aligned with the first
        axis of ``tensor``.

    Returns
    -------
    Tensor
        ``mask`` tiled to the shape of ``tensor``.

    See also
    --------
    :func:`apply_mask`
    """
    # TODO: require axis to be ordered as in `orient_and_conform`.
    # Ideally, we should create a common underlying function for
    # the shared parts of both operations (i.e., identifying
    # aligning vs. expanding axes).
    tensor = jnp.asarray(tensor)
    if batch and tensor.ndim == 1:
        batch = False
    if isinstance(axis, int):
        if not batch:
            shape_pfx = tensor.shape[:axis]
            mask = jnp.tile(mask, (*shape_pfx, 1))
            return mask
        axis = (axis,)
    if batch:
        axis = (0, *axis)
    # TODO: this feels like it will produce unexpected behaviour.
    mask = mask.squeeze()
    tile = list(tensor.shape)
    shape = [1 for _ in range(tensor.ndim)]
    for i, ax in enumerate(axis):
        tile[ax] = 1
        shape[ax] = mask.shape[i]
    mask = jnp.tile(mask.reshape(*shape), tile)
    return mask


def apply_mask(
    tensor: Tensor,
    msk: Tensor,
    axis: int,
) -> Tensor:
    """
    Mask a tensor along an axis.

    .. warning::

        Only one-dimensional masks are supported. For multi-dimensional masks,
        use :func:`conform_mask`.

    .. warning::

        Discouraged: incompatible with ``jax.jit``, because the output shape
        depends on the mask.

    Parameters
    ----------
    tensor
        Tensor to mask.
    msk
        Boolean mask over ``axis``.
    axis
        Axis to mask.

    Returns
    -------
    Tensor
        ``tensor`` with the entries along ``axis`` where ``msk`` is False
        removed.

    See also
    --------
    :func:`conform_mask`
    :func:`mask_tensor`
    """
    tensor = jnp.asarray(tensor)
    shape_pfx = tensor.shape[:axis]
    if axis == -1:
        shape_sfx = ()
    else:
        shape_sfx = tensor.shape[(axis + 1) :]
    msk = jnp.tile(msk, (*shape_pfx, 1))
    return tensor[msk].reshape(*shape_pfx, -1, *shape_sfx)


def mask_tensor(
    tensor: Tensor,
    mask: Tensor,
    axis: Sequence[int],
    fill_value: Union[float, Tensor] = 0,
):
    mask = conform_mask(tensor=tensor, mask=mask, axis=axis)
    return jnp.where(mask, tensor, fill_value)


def complex_decompose(complex: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Decompose a complex-valued tensor into amplitude and phase components.

    :Dimension:
        Each output is of the same shape as the input.

    Parameters
    ----------
    complex
        Complex-valued tensor.

    Returns
    -------
    ampl : Tensor
        Amplitude of each entry in the input tensor.
    phase : Tensor
        Phase of each entry in the input tensor, in radians.

    See also
    --------
    :func:`complex_recompose`
    """
    ampl = jnp.abs(complex)
    phase = jnp.angle(complex)
    return ampl, phase


def complex_recompose(ampl: Tensor, phase: Tensor) -> Tensor:
    """
    Compose a complex tensor from its amplitude and phase.

    :Dimension:
        Both inputs must be the same shape (or broadcastable). The
        output is the same shape as the inputs.

    Parameters
    ----------
    ampl
        Real-valued array storing complex number amplitudes.
    phase
        Real-valued array storing complex number phases in radians.

    Returns
    -------
    complex : Tensor
        Complex numbers formed from the specified amplitudes and phases.

    See also
    --------
    :func:`complex_decompose`
    """
    # TODO : consider using the complex exponential function,
    # depending on the gradient properties
    # return ampl * jnp.exp(phase * 1j)
    return ampl * (jnp.cos(phase) + 1j * jnp.sin(phase))


def amplitude_apply(func: Callable) -> Callable:
    """
    Return ``func`` applied to the amplitude of a complex tensor.
    """

    def wrapper(complex: Tensor) -> Tensor:
        ampl, phase = complex_decompose(complex)
        return complex_recompose(func(ampl), phase)

    return wrapper
