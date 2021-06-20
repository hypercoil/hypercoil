# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Delta-plus initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~
Initialise parameters as a set of delta functions, plus Gaussian noise.
"""
import torch
from functools import partial
from ..functional.domainbase import Identity
from .base import BaseInitialiser


def deltaplus_init_(tensor, loc=None, scale=None, var=0.2, domain=None):
    """
    Delta-plus initialisation.

    Initialise a tensor as a delta function added to Gaussian noise.

    Parameters
    ----------
    tensor : Tensor
        Tensor to initialise in-place.
    loc : tuple or None (default None)
        Location of the delta function in array coordinates.
    scale : float or None (default None)
        Height of the delta function.
    var : float
        Variance of the Gaussian distribution from which the random noise is
        sampled.
    domain : Domain object (default Identity)
        Used in conjunction with an activation function to constrain or
        transform the values of the initialised tensor. For instance, using
        the Atanh domain with default scale constrains the tensor as seen by
        data to the range of the tanh function, (-1, 1). Domain objects can
        be used with compatible modules and are documented further in
        `hypercoil.functional.domain`. If no domain is specified, the Identity
        domain is used, which does not apply any transformations or
        constraints.

    Returns
    -------
    None. The input tensor is initialised in-place.
    """
    domain = domain or Identity()
    loc = loc or tuple([x // 2 for x in tensor.size()])
    scale = scale or 1
    val = torch.zeros_like(tensor)
    val[(...,) + loc] += scale
    val = domain.preimage(val)
    val += torch.randn(tensor.size()) * var
    val.type(tensor.dtype)
    tensor.copy_(val)


class DeltaPlusInit(BaseInitialiser):
    def __init__(self, loc=None, scale=None, var=0.2, domain=None):
        init = partial(deltaplus_init_, loc=loc, scale=scale,
                       var=var, domain=domain)
        super(DeltaPlusInit, self).__init__(init=init)
