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
    dim = tensor.size()[-2:]
    val = toeplitz(c=c, r=r, dim=dim, fill_value=fill_value)
    val.type(tensor.dtype)
    tensor[:] = val
