# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Residualise
~~~~~~~~~~~
Residualise tensor block via least squares. No parameters here.
"""
import torch
from torch.nn import Module
from ..functional.resid import residualise


#TODO: assess backprop properties of this approach vs conditional correlation
class Residualise(Module):
    def __init__(self, rowvar=True, driver='gelsd'):
        super(Residualise, self).__init__()

        self.rowvar = rowvar
        self.driver = driver

    def forward(self, Y, X, mask=None):
        if mask is not None:
            Y = mask * Y
            X = mask * X
        return residualise(
            Y=Y, X=X, rowvar=self.rowvar, driver=self.driver
        )
