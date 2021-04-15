# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Initialiser
~~~~~~~~~~~
Base initialiser for a module.
"""
import torch


def uniform_init_(tensor, min=0, max=1):
    val = torch.rand_like(tensor) * (max - min) + min
    tensor[:] = val


class DomainInitialiser(object):
    def __init__(self, init=None, domain=None):
        self.init = init or uniform_init_
        self.domain = domain or Identity()

    def __call__(self, tensor):
        rg = tensor.requires_grad
        tensor.requires_grad = False
        self.init(tensor)
        tensor[:] = self.domain.preimage(tensor)
        tensor.requires_grad = rg


class BaseInitialiser(DomainInitialiser):
    def __init__(self, init=None):
        super(BaseInitialiser, self).__init__(init=init, domain=None)
