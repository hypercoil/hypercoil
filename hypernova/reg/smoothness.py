# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Smoothness
~~~~~~~~~~
Penalise backwards differences to favour smoothness.
"""
from torch import diff, norm as pnorm
from torch.nn import Module
from functools import partial


class SmoothnessPenalty(Module):
    def __init__(self, nu=1, axis=-1, prepend=None, append=None, norm=None):
        self.nu = nu
        self.axis = axis
        self.prepend = prepend
        self.append = append
        self.norm = norm or partial(pnorm, p=2)

    def forward(self, weight):
        return self.nu * self.norm(
            diff(weight, dim=self.axis,
                 prepend=self.prepend,
                 append=self.append)
        )
