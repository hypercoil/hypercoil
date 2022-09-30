# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Crosshair kernel
~~~~~~~~~~~~~~~~
Elementary operations over a crosshair kernel.
"""
from __future__ import annotations
from typing import Callable, Tuple

import jax.numpy as jnp

from ..engine import NestedDocParse, Tensor


def document_crosshair(f: Callable) -> Callable:
    crosshair_inner_long = """
    For each entry of the input matrices :math:`X_{ij}`, the crosshair product
    is the scalar sum over elements in either the same row or column as the
    entry multiplied with the corresponding element of the other input matrix.
    The product dimensions are accordingly equal to the dimensions of the input
    matrices."""
    crosshair_norm_long = """
    For each entry of the input matrices :math:`X_{ij}`, the crosshair norm
    is the norm over a vector containing all elements in either the same row or
    column as the entry. The norm dimensions are accordingly equal to the
    dimensions of the input matrices."""
    crosshair_dim_spec = """
    :Dimension:
    The input tensors must have at least two axes and must have the same shape.
    The dimension of the output tensor exactly equals the dimension of the
    input tensor."""
    crosshair_unary_pparams = """
    A: Tensor
        Input tensor."""
    crosshair_binary_pparams = """
    A: Tensor
        First input tensor.
    B: Tensor
        Second input tensor."""
    crosshair_rc_spec = """
    row: int
        Axis of the input tensors over which the row index increases.
    col: int
        Axis of the input tensors over which the column index increases."""
    crosshair_return_spec = """
    Returns
    -------
    Tensor
        Tensor in which each entry contains the result of the computation
        over a crosshair-shaped kernel."""
    fmt = NestedDocParse(
        crosshair_inner_long=crosshair_inner_long,
        crosshair_norm_long=crosshair_norm_long,
        crosshair_dim_spec=crosshair_dim_spec,
        crosshair_unary_pparams=crosshair_unary_pparams,
        crosshair_binary_pparams=crosshair_binary_pparams,
        crosshair_rc_spec=crosshair_rc_spec,
        crosshair_return_spec=crosshair_return_spec,
    )
    f.__doc__ = f.__doc__.format_map(fmt)
    return f


@document_crosshair
def crosshair_dot(
    A: Tensor,
    B: Tensor,
    row: int = -2,
    col: int = -1,
) -> Tensor:
    """
    Local dot product between two matrices over a crosshair kernel.
    \
    {crosshair_inner_long}
    \
    {crosshair_dim_spec}

    Parameters
    ----------\
    {crosshair_binary_pparams}\
    {crosshair_rc_spec}
    \
    {crosshair_return_spec}

    See also
    --------
    crosshair_dot_gen: Extend the kernel over more than 2 dimensions.
    """
    prod = A * B
    rows = prod.sum(row, keepdims=True)
    cols = prod.sum(col, keepdims=True)
    return rows + cols - prod


@document_crosshair
def crosshair_norm_l2(
    A: Tensor,
    row: int = -2,
    col: int = -1,
) -> Tensor:
    """
    Compute the local L2 norm on a matrix over a crosshair kernel.
    \
    {crosshair_norm_long}
    \
    {crosshair_dim_spec}

    Parameters
    ----------\
    {crosshair_unary_pparams}\
    {crosshair_rc_spec}
    \
    {crosshair_return_spec}
    """
    return jnp.sqrt(crosshair_dot(A, A, row=row, col=col))


@document_crosshair
def crosshair_norm_l1(
    A: Tensor,
    row: int = -2,
    col: int = -1,
) -> Tensor:
    """
    Compute the local L1 norm on a matrix over a crosshair kernel.
    \
    {crosshair_norm_long}
    \
    {crosshair_dim_spec}

    Parameters
    ----------\
    {crosshair_unary_pparams}\
    {crosshair_rc_spec}
    \
    {crosshair_return_spec}
    """
    abs = jnp.abs(A)
    rows = abs.sum(row, keepdims=True)
    cols = abs.sum(col, keepdims=True)
    return rows + cols - abs


# TODO: marking this as an experimental function
def crosshair_dot_gen(
    A: Tensor,
    B: Tensor,
    axes: Tuple[int, int] = (-2, -1),
) -> Tensor:
    """
    A generalised version of the crosshair dot product where the crosshair can
    be extended over any number of dimensions. As it suffers poor performance
    relative to the 2-D implementation and as its use cases are likely narrow,
    its correctness has not been tested.
    """
    prod = A * B
    repeats = len(axes) - 1
    axis_sum = [None for _ in axes]
    for i, ax in enumerate(axes):
        axis_sum[i] = prod.sum(ax, keepdims=True)
    out = axis_sum[0]
    for a in axis_sum[1:]:
        out = out + a
    return out - repeats * prod
