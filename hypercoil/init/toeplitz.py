# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Toeplitz initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~
Initialise parameters as a stack of Toeplitz-structured banded matrices.
"""
import torch
from functools import partial
from .base import BaseInitialiser
from ..functional import toeplitz
from .domain import Identity


def toeplitz_init_(tensor, c, r=None, fill_value=0, domain=None):
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
    domain : Domain object (default Identity)
        Used in conjunction with an activation function to constrain or
        transform the values of the initialised tensor. For instance, using
        the Atanh domain with default scale constrains the tensor as seen by
        data to the range of the tanh function, (-1, 1). Domain objects can
        be used with compatible modules and are documented further in
        `hypercoil.init.domain`. If no domain is specified, the Identity
        domain is used, which does not apply any transformations or
        constraints.

    Returns
    -------
    None. The input tensor is initialised in-place.
    """
    domain = domain or Identity()
    dim = tensor.size()[-2:]
    val = toeplitz(
        c=c, r=r, dim=dim,
        fill_value=fill_value,
        dtype=tensor.dtype,
        device=tensor.device
    )
    val = domain.preimage(val)
    tensor.copy_(val)


class ToeplitzInit(BaseInitialiser):
    def __init__(self, c, r=None, fill_value=0, domain=None):
        init = partial(toeplitz_init_, c=c, r=r,
                       fill_value=fill_value, domain=domain)
        super(ToeplitzInit, self).__init__(init=init)
        self.fill_value = fill_value
        self.domain = domain
