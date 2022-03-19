# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base regularisations
~~~~~~~~~~~~~~~~~~~~
Base modules for regularisation classes.
"""
from torch.nn import Module


class LossApply(Module):
    def __init__(self, loss, apply=None):
        super(LossApply, self).__init__()
        if apply is None:
            apply = lambda x: x
        self.loss = loss
        self.apply = apply

    def __repr__(self):
        return f'[ν = {self.loss.nu}]{type(self.loss).__name__}'

    def forward(self, *args, **kwargs):
        return self.loss(self.apply(*args, **kwargs))


class ReducingRegularisation(Module):
    def __init__(self, nu, reduction, reg):
        super(ReducingRegularisation, self).__init__()
        self.nu = nu
        self.reduction = reduction
        self.reg = reg

    def __repr__(self):
        return f'[ν = {self.nu}]{type(self).__name__}'

    def forward(self, *args, **kwargs):
        return self.nu * self.reduction(self.reg(*args, **kwargs))
