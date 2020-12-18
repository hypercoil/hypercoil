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
    cone_project_spd, tangent_project_spd, mean_euc_spd
)


class _TangentProject(Module):
    def __init__(self, mean=None, recondition=0):
        super(_TangentProject, self).__init__()
        self.dest = 'tangent'
        self.mean = mean or mean_euc_spd
        self.recondition = recondition

    def extra_repr(self):
        s = f'mean={self.mean.__name__}'
        if self.recondition != 0:
            s += f', psi={self.recondition}'
        return s

    def forward(self, input, dest=None):
        dest = dest or self.dest
        weight = self._weight(input)
        if dest == 'tangent':
            return tangent_project_spd(input, weight, self.recondition)
        elif dest == 'cone':
            return cone_project_spd(input, weight, self.recondition)


class TangentProject(_TangentProject):
    def __init__(self, init_data, mean=None, recondition=0):
        super(TangentProject, self).__init__(mean, recondition)
        self.weight = Parameter(torch.Tensor(
            *init_data.size()[1:]
        ))
        self.reset_parameters(init_data)

    def _weight(self, input):
        return self.weight

    def reset_parameters(self, init_data):
        rg = self.weight.requires_grad
        self.weight.requires_grad = False
        self.weight[:] = self.mean(init_data, recondition=self.recondition)
        self.weight.requires_grad = rg


class BatchTangentProject(_TangentProject):
    def _weight(self, input):
        return self.mean(input, recondition=self.recondition)
