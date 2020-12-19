# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Positive semidefinite cone
~~~~~~~~~~~~~~~~~~~~~~~~~~
Initialise and compute means and mean blocks in the positive semidefinite cone.
"""
import torch
from torch.nn import Module, Parameter, init
from ..functional import (
    mean_euc_spd, mean_harm_spd, mean_logeuc_spd, mean_geom_spd, SPSDNoiseSource
)


def mean_block_spd(mean_specs, init_data):
    return torch.stack([spec(init_data) for spec in mean_specs]).squeeze(0)


def mean_apply_block(mean_specs, data):
    return torch.stack([spec(d) for spec, d in zip(mean_specs, data)])


def tangency_init_(tensor, mean_specs, init_data, std=0):
    rg = tensor.requires_grad
    tensor.requires_grad = False
    means = mean_block_spd(mean_specs, init_data)
    if std > 0:
        means = SPSDNoiseSource(std=std).inject(means)
    tensor[:] = means
    tensor.requires_grad = rg


class _SemidefiniteMean(Module):
    def __init__(self, axis=0):
        super(_SemidefiniteMean, self).__init__()
        self.axis = axis

    def extra_repr(self):
        return f'axis={self.axis}'


class SPDEuclideanMean(_SemidefiniteMean):
    def forward(self, input):
        return mean_euc_spd(input, axis=self.axis)


class SPDHarmonicMean(_SemidefiniteMean):
    def forward(self, input):
        return mean_harm_spd(input, axis=self.axis)


class SPDLogEuclideanMean(_SemidefiniteMean):
    def forward(self, input):
        return mean_logeuc_spd(input, axis=self.axis)


class SPDGeometricMean(_SemidefiniteMean):
    def __init__(self, axis=0, psi=0, eps=1e-6, max_iter=10):
        super(SPDGeometricMean, self).__init__(axis=axis)
        self.psi = psi
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, input):
        return mean_geom_spd(
            input, axis=self.axis, recondition=self.psi,
            eps=self.eps, max_iter=self.max_iter
        )

    def extra_repr(self):
        s = super(SPDGeometricMean, self).extra_repr()
        if self.psi > 0:
            s += f', psi={self.psi}'
        s += f', max_iter={self.max_iter}'
        return s
