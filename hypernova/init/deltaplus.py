# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Delta-plus initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~
Initialise parameters as a set of delta functions, plus Gaussian noise.
"""
import torch


def deltaplus_init_(tensor, loc=None, scale=None, var=0.2):
    """
    Delta-plus initialisation.

    Initialise a tensor as a delta function added to Gaussian noise.

    Parameters
    ----------
    tensor : Tensor
        Tensor to initialise in-place.
    loc : iterable or None (default None)
        Location of the delta function in array coordinates.
    scale : float or None (default None)
        Height of the delta function.
    std : float
        Variance of the Gaussian distribution from which the random noise is
        sampled.

    Returns
    -------
    None. The input tensor is initialised in-place.
    """
    loc = loc or tuple([x // 2 for x in tensor.size()])
    scale = scale or 1
    val = torch.zeros_like(tensor)
    val[loc] += scale
    val += torch.randn(tensor.size()) * std
    val.type(tensor.dtype)
    tensor[:] = val
