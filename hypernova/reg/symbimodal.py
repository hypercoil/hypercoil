# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Symmetric bimodal
~~~~~~~~~~~~~~~~~
Symmetric bimodal regularisations using absolute value.
"""
from torch import norm as pnorm
from torch.nn import Module
from functools import partial


def symmetric_bimodal_distance(weight, modes=(0, 1)):
    mean = sum(modes) / 2
    step = modes[1] - mean
    return (weight - mean).abs() - step


class SymmetricBimodal(Module):
    def __init__(self, nu=1, modes=(0, 1), norm=None):
        self.nu = nu
        self.modes = modes
        self.norm = norm or partial(pnorm, p=2)

    def forward(self, weight):
        return self.nu * self.norm(
            symmetric_bimodal_distance(weight, self.modes)
        )
