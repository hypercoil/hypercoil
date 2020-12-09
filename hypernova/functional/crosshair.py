# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Crosshair kernel
~~~~~~~~~~~~~~~~
Elementary operations over a crosshair kernel.
"""
import torch


def crosshair_dot(A, B, row=-2, col=-1):
    """
    Compute the local dot product between two matrices over a crosshair
    kernel.

    Dimension
    ---------

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
    """
    prod = A * B
    rows = prod.sum(row, keepdim=True)
    cols = prod.sum(col, keepdim=True)
    return rows + cols - prod


def crosshair_norm_l2(A, row=-2, col=-1):
    """
    Compute the local L2 norm on a matrix over a crosshair kernel.

    Parameters
    ----------
    A: Tensor
        Input tensor.
    row: int
        Axis of the input tensor over which the row index increases.
    col: int
        Axis of the input tensor over which the column index increases.
    """
    return torch.sqrt(crosshair_dot(A, A, row=row, col=col))


def crosshair_norm_l1(A, row=-2, col=-1):
    """
    Compute the local L1 norm on a matrix over a crosshair kernel.

    Parameters
    ----------
    A: Tensor
        Input tensor.
    row: int
        Axis of the input tensor over which the row index increases.
    col: int
        Axis of the input tensor over which the column index increases.
    """
    abs = torch.abs(A)
    rows = abs.sum(row).unsqueeze(row)
    cols = abs.sum(col).unsqueeze(col)
    return rows + cols - abs
