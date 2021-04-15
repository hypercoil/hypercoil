# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Symmetric bimodal
~~~~~~~~~~~~~~~~~
Symmetric bimodal regularisations using absolute value.
"""
from functools import partial
from .norm import NormedRegularisation



def symmetric_bimodal_distance(weight, modes=(0, 1)):
    mean = sum(modes) / 2
    step = max(modes) - mean
    return ((weight - mean).abs() - step).abs()


class SymmetricBimodal(NormedRegularisation):
    def __init__(self, nu=1, modes=(0, 1), norm=2):
        reg = partial(
            symmetric_bimodal_distance,
            modes=modes
        )
        super(SymmetricBimodal, self).__init__(nu=nu, p=norm, reg=reg)
