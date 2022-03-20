# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base losses
~~~~~~~~~~~
Base modules for loss classes.
"""
from torch.nn import Module


class Loss(Module):
    @property
    def extra_repr(self):
        return ()

    def __repr__(self):
        if self.extra_repr:
            s = ', '.join((f'ν = {self.nu}', *self.extra_repr()))
            return f'[{s}]{self.name}'
        return f'[ν = {self.nu}]{self.name}'


class LossApply(Module):
    def __init__(self, loss, apply=None):
        super(LossApply, self).__init__()
        if apply is None:
            apply = lambda x: x
        self.loss = loss
        self.apply = apply

    def __repr__(self):
        return self.loss.__repr__()

    def forward(self, *args, **kwargs):
        return self.loss(self.apply(*args, **kwargs))


class ReducingLoss(Loss):
    def __init__(self, nu, reduction, loss, name=None):
        super(ReducingLoss, self).__init__()
        if name is None:
            name = type(self).__name__
        self.nu = nu
        self.reduction = reduction
        self.loss = loss
        self.name = name

    def forward(self, *args, **kwargs):
        return self.nu * self.reduction(self.loss(*args, **kwargs))
