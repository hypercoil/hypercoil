# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
SPD Noise
~~~~~~~~~
Modules that inject symmetric positive (semi)definite noise.
"""
import torch
from torch.nn import Module, Parameter, init
from ..functional import SPSDNoiseSource
from ..functional.cov import corrnorm


class SPDNoise(Module):
    def __init__(self, std, norm=True):
        super(SPDNoise, self).__init__()
        self.norm = norm
        self.noise = SPSDNoiseSource(std=std)

    def train(self, mode=True):
        super(_Cov, self).train(mode)
        self.noise.train(mode)

    def eval(self):
        super(_Cov, self).eval()
        self.noise.eval()

    def forward(self, input):
        x = self.noise.inject(input)
        if self.norm:
            return x / corrnorm(x)
        else:
            return x
