# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Positive semidefinite cone
~~~~~~~~~~~~~~~~~~~~~~~~~~
Modules that project data between the positive semidefinite cone proper
subspaces tangent to the cone.
"""
import torch
from torch.nn import Module, Parameter, init
from ..functional import (
    cone_project_spd, tangent_project_spd, invert_spd, mean_euc_spd,
    mean_harm_spd, mean_logeuc_spd, mean_geom_spd, SPSDNoiseSource
)
from ..init.semidefinite import mean_block_spd, mean_apply_block, tangency_init_


class _TangentProject(Module):
    def __init__(self, mean_specs=None, recondition=0):
        super(_TangentProject, self).__init__()
        self.dest = 'tangent'
        self.mean_specs = mean_specs or [SPDEuclideanMean()]
        self.out_channels = len(self.mean_specs)
        self.recondition = recondition

    def extra_repr(self):
        s = ',\n'.join([f'(mean) {spec.__repr__()}'
                         for spec in self.mean_specs])
        if self.recondition != 0:
            s += f',\npsi={self.recondition}'
        if self.out_channels > 1:
            s += f',\nout_channels={self.out_channels}'
        return s

    def forward(self, input, dest=None):
        if self.out_channels > 1:
            input = input.unsqueeze(-3)
        dest = dest or self.dest
        if dest == 'tangent':
            return tangent_project_spd(input, self.weight, self.recondition)
        elif dest == 'cone':
            return cone_project_spd(input, self.weight, self.recondition)


class TangentProject(_TangentProject):
    def __init__(self, init_data, mean_specs=None, recondition=0, std=0):
        super(TangentProject, self).__init__(mean_specs, recondition)
        if self.out_channels > 1:
            self.weight = Parameter(torch.Tensor(
                *init_data.size()[1:-2],
                self.out_channels,
                init_data.size(-2),
                init_data.size(-1),
            ))
        else:
            self.weight = Parameter(torch.Tensor(
                *init_data.size()[1:]
            ))
        self.reset_parameters(init_data, std)

    def reset_parameters(self, init_data, std=0):
        tangency_init_(self.weight, self.mean_specs, init_data, std)


class BatchTangentProject(_TangentProject):
    def __init__(self, mean_specs=None, recondition=0, inertia=0):
        super(BatchTangentProject, self).__init__(mean_specs, recondition)
        self.inertia = inertia
        self.weight = None

    def forward(self, input, dest=None):
        if dest != 'cone':
            weight = mean_block_spd(self.mean_specs, input)
            if self.weight is None:
                self.weight = weight.detach()
            self.weight = (
                self.inertia * self.weight + (1 - self.inertia) * weight
            ).detach()
        elif self.weight is None:
            ref = invert_spd(torch.matrix_exp(input.mean(0)))
            self.weight = torch.stack(self.out_channels * [ref])
            print(self.weight.shape)
        out = super(BatchTangentProject, self).forward(input, dest)

        if dest == 'cone':
            if self.out_channels > 1:
                weight = mean_apply_block(self.mean_specs, out)
            else:
                weight = self.mean_specs[0](out)
            print(weight.shape)
            self.weight = (
                self.inertia * self.weight + (1 - self.inertia) * weight
            ).detach()
            print(self.weight.shape)
        return out
