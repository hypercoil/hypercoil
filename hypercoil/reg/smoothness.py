# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Smoothness
~~~~~~~~~~
Penalise backwards differences to favour smoothness.
"""
from torch import diff
from functools import partial
from .norm import NormedRegularisation


class SmoothnessPenalty(NormedRegularisation):
    def __init__(self, nu=1, axis=-1, prepend=None, append=None, norm=2):
        reg = partial(
            diff, dim=axis, prepend=prepend, append=append
        )
        super(SmoothnessPenalty, self).__init__(nu=nu, p=norm, reg=reg)

    def extra_repr(self):
        return f'nu={self.nu}, norm=L{self.p}'
