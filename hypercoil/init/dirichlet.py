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
from ..functional.domain import MultiLogit


def dirichlet_init_(tensor, distr, axis=-1):
    dim = list(tensor.size())
    del(dim[axis])
    val = distr.sample(dim)
    val = torch.movedim(val, -1, axis)
    tensor.copy_(val)


class DirichletInit(DomainInitialiser):
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
