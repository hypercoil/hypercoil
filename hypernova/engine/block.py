# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Diff block
~~~~~~~~~~
Differentiable program block. Currently minimal functionality.
"""
from torch.nn import Module


class Block(Module):
    def __init__(self, module, init=None, reg=None, loss=None):
        self.module = module
        self.init = init
        self.reg = reg or []
        self.loss = loss or []

    def init(self, *args, data=None, **kwargs):
        if data is not None:
            self.module = self.module(init=self.init, data=data)
        else:
            self.module = self.module(init=self.init)

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        reg = sum([r(self.module.weight) for r in self.reg])
        loss = sum([l(out) for l in self.loss])
        return out, reg, loss
