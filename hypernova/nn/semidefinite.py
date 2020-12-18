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


class TangentProject(Module):
    def __init__(self, init_data, mean=None, recondition=0):
        super(TangentProject, self).__init__()
        self.mean = mean or mean_euc_spd
        self.recondition = recondition
        self.dest = 'tangent'
        self.weight = Parameter(torch.Tensor(
            *init_data.size()[1:]
        ), requires_grad=False)

        self.reset_parameters(init_data)

    def reset_parameters(self, init_data):
        rg = self.weight.requires_grad
        self.weight.requires_grad = False
        self.weight[:] = self.mean(init_data)
        self.weight.requires_grad = rg

    def extra_repr(self):
        s += f'init={self.mean.__name__}'
        if self.recondition != 0:
            s += f', psi={self.recondition}'
        return s

    def forward(self, input, dest=self.dest):
        if dest == 'tangent':
            return tangent_project_spd(input, self.weight, self.recondition)
        elif dest == 'cone':
            return cone_project_spd(input, self.weight, self.recondition)
