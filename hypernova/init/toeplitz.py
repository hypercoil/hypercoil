# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Toeplitz initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~
Initialise parameters as a stack of Toeplitz-structured banded matrices.
"""
import torch
from ..functional import toeplitz


def toeplitz_init_(tensor, c, r=None, fill_value=0):
    """
    Initialise a tensor as a stack of banded matrices with Toeplitz structure.

    Parameters
    ----------
    c: Tensor
        Tensor of entries in the first column of each Toeplitz matrix. The
        first axis corresponds to a single matrix column; additional dimensions
        correspond to concatenation of Toeplitz matrices into a stack or block
        tensor.
    r: Tensor or None (default None)
        Tensor of entries in the first row of each Toeplitz matrix. The first
        axis corresponds to a single matrix row; additional dimensions
        correspond to concatenation of Toeplitz matrices into a stack or block
        tensor. The first entry in each column should be the same as the first
        entry in the corresponding column of `c`; otherwise, it will be
        ignored. If this is None, then a symmetric Toeplitz matrix will be
        created using the same tensor for both the row and the column.
    fill_value: Tensor or float (default 0)
        Specifies the value that should be used to populate the off-diagonals
        of each Toeplitz matrix if the specified row and column elements are
        extended to conform with the specified `dim`. If this is a tensor, then
        each entry corresponds to the fill value in a different data channel.
        Has no effect if `dim` is None.

    Returns
    -------
    None. The input tensor is initialised in-place.
    """
    dim = tensor.size()[-2:]
    val = toeplitz(c=c, r=r, dim=dim, fill_value=fill_value)
    val.type(tensor.dtype)
    tensor[:] = val
