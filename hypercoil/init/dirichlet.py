# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Dirichlet initialiser
~~~~~~~~~~~~~~~~~~~~~
Initialise a tensor such that elements along a given axis are Dirichlet
samples.
"""
import torch
from functools import partial
from torch.distributions.dirichlet import Dirichlet
from .base import DomainInitialiser
from .domain import MultiLogit


def dirichlet_init_(tensor, distr, axis=-1):
    """
    Dirichlet sample initialisation.

    Initialise a tensor such that any 1D slice through that tensor along a
    given axis is a sample from a specified Dirichlet distribution. Each 1D
    slice can therefore be understood as encoding a categorical probability
    distribution.

    Parameters
    ----------
    tensor : Tensor
        Tensor to initialise in-place.
    distr : instance of torch.distributions.Dirichlet
        Parametrised Dirichlet distribution from which all 1D slices of the
        input tensor along the specified axis are sampled.
    axis : int (default -1)
        Axis along which slices are sampled from the specified Dirichlet
        distribution.

    Returns
    -------
    None. The input tensor is initialised in-place.
    """
    dim = list(tensor.size())
    del(dim[axis])
    val = distr.sample(dim).to(dtype=tensor.dtype, device=tensor.device)
    val = torch.movedim(val, -1, axis)
    tensor.copy_(val)


class DirichletInit(DomainInitialiser):
    """
    Dirichlet sample initialiser.

    Initialise a tensor such that any 1D slice through that tensor along a
    given axis is a sample from a specified Dirichlet distribution. Each 1D
    slice can therefore be understood as encoding a categorical probability
    distribution.

    When this initialiser is coupled with a softmax domain (`MultiLogit`),
    the parent module can ensure that 1D slices of the initialised weights
    remain in the probability simplex as the module learns. This is currently
    the default behaviour.

    Parameters
    ----------
    n_classes : int
        Number of classes in the distribution.
    concentration : iterable
        Concentration parameter for the Dirichlet distribution. This must have
        length equal to `n_classes`.
    axis : int (default -1)
        Axis along which slices are sampled from the specified Dirichlet
        distribution.
    domain : Domain object (default MultiLogit)
        Used in conjunction with an activation function to constrain or
        transform the values of the initialised tensor. For instance, using
        the MultiLogit domain constrains slices of the tensor (as seen by
        data) to lie in the appropriate probability simplex. Domain objects
        can be used with compatible modules and are documented further in
        `hypercoil.init.domain`. If no domain is specified, the
        MultiLogit domain is used.
    """
    def __init__(self, n_classes, concentration=None, axis=-1, domain=None):
        if isinstance(concentration, torch.Tensor):
            self.concentration = concentration
        else:
            self.concentration = (
                concentration or
                torch.tensor([10.0] * n_classes)
            )
        assert len(self.concentration) == n_classes
        self.distr = Dirichlet(concentration=self.concentration)
        self.axis = axis
        self.init = partial(dirichlet_init_, distr=self.distr, axis=self.axis)
        self.domain = domain or MultiLogit(axis=self.axis)
