# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Crosshair kernel
~~~~~~~~~~~~~~~~
Elementary operations over a crosshair kernel.
"""
import jax.numpy as jnp


def crosshair_dot(A, B, row=-2, col=-1):
    """
    Local dot product between two matrices over a crosshair kernel.

    For each entry of the input matrices :math:`X_{ij}`, the crosshair product
    is the scalar sum over elements in either the same row or column as the
    entry multiplied with the corresponding element of the other input matrix.
    The product dimensions are accordingly equal to the dimensions of the input
    matrices.

    :Dimension:
    The input tensors must have at least two axes and must have the same shape.
    The dimension of the output tensor exactly equals the dimension of the
    input tensor.

    Parameters
    ----------
    A: Tensor
        First input tensor.
    B: Tensor
        Second input tensor.
    row: int
        Axis of the input tensors over which the row index increases.
    col: int
        Axis of the input tensors over which the column index increases.

    Returns
    -------
    output: Tensor
        Tensor in which each entry contains the inner product between entries
        of the two input matrices computed over a crosshair-shaped kernel.

    See also
    --------
    crosshair_dot_gen: Extend the kernel over more than 2 dimensions.
    """
    prod = A * B
    rows = prod.sum(row, keepdims=True)
    cols = prod.sum(col, keepdims=True)
    return rows + cols - prod


def crosshair_norm_l2(A, row=-2, col=-1):
    """
    Compute the local L2 norm on a matrix over a crosshair kernel.

    For each entry of the input matrices :math:`X_{ij}`, the crosshair norm
    is the norm over a vector containing all elements in either the same row or
    column as the entry. The norm dimensions are accordingly equal to the
    dimensions of the input matrices.

    :Dimension:
    The input tensors must have at least two axes and must have the same shape.
    The dimension of the output tensor exactly equals the dimension of the
    input tensor.

    Parameters
    ----------
    A: Tensor
        Input tensor.
    row: int
        Axis of the input tensor over which the row index increases.
    col: int
        Axis of the input tensor over which the column index increases.

    Returns
    -------
    output: Tensor
        Tensor in which each entry contains the norm of the entries of the
        input matrix computed over a crosshair-shaped kernel.
    """
    return jnp.sqrt(crosshair_dot(A, A, row=row, col=col))


def crosshair_norm_l1(A, row=-2, col=-1):
    """
    Compute the local L1 norm on a matrix over a crosshair kernel.

    For each entry of the input matrices :math:`X_{ij}`, the crosshair norm
    is the norm over a vector containing all elements in either the same row or
    column as the entry. The norm dimensions are accordingly equal to the
    dimensions of the input matrices.

    :Dimension:
    The input tensors must have at least two axes and must have the same shape.
    The dimension of the output tensor exactly equals the dimension of the
    input tensor.

    Parameters
    ----------
    A: Tensor
        Input tensor.
    row: int
        Axis of the input tensor over which the row index increases.
    col: int
        Axis of the input tensor over which the column index increases.

    Returns
    -------
    output: Tensor
        Tensor in which each entry contains the norm of the entries of the
        input matrix computed over a crosshair-shaped kernel.
    """
    abs = jnp.abs(A)
    rows = abs.sum(row, keepdims=True)
    cols = abs.sum(col, keepdims=True)
    return rows + cols - abs


#TODO: marking this as an experimental function
def crosshair_dot_gen(A, B, axes=(-2, -1)):
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
