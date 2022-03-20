# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Smoothness
~~~~~~~~~~
Penalise backwards differences to favour smoothness.
"""
import torch
from functools import partial
from .norm import NormedLoss


class SmoothnessPenalty(NormedLoss):
    def __init__(self, nu=1, axis=-1, prepend=None,
                 append=None, norm=2, name=None):
        loss = partial(
            torch.diff, dim=axis, prepend=prepend, append=append
        )
        super(SmoothnessPenalty, self).__init__(
            nu=nu,
            p=norm,
            loss=loss,
            axis=axis,
            name=name
        )
