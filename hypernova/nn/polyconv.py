# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Polynomial convolution
~~~~~~~~~~~~~~~~~~~~~~
Modules supporting polynomial convolution of time series and other data.
"""
import math
import torch
from torch import nn
from torch.nn import Module, Parameter, init
from ..functional import polyconv2d
from ..init.laplace import laplace_init_


class PolyConv2D(Module):
    def __init__(self, degree=2, out_channels=1, memory=3, kernel_width=1,
                 padding=None, bias=False, include_const=False,
                 future_sight=False, init_=laplace_init_, init_params=None):
        super(PolyConv2D, self).__init__()

        if init_params is None:
            init_params = {}
        if 'loc' not in init_params:
            if include_const:
                init_params['loc'] = (1, 0, memory)
            else:
                init_params['loc'] = (0, 0, memory)

        self.in_channels = degree + include_const
        self.out_channels = out_channels
        self.memory = memory
        self.kernel_length = 2 * memory + 1
        self.kernel_width = kernel_width
        self.padding = padding
        self.degree = degree
        self.include_const = include_const
        self.future_sight = future_sight
        self.init_ = init_
        self.init_params = init_params

        self.weight = Parameter(torch.Tensor(
            self.out_channels,
            self.in_channels,
            self.kernel_width,
            self.kernel_length))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.init_(self.weight, **self.init_params)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

        if self.future_sight:
            self.mask = None
        else:
            self.mask = Parameter(torch.ones_like(self.weight))
            self.mask.requires_grad = False
            self.weight.requires_grad = False
            self.mask[:, :, :, (self.memory + 1):] = 0
            self.weight[:] = self.weight * self.mask
            self.weight.requires_grad = True

    def __repr__(self):
        s = (
            f'{self.__class__.__name__}(degree={self.degree}, out_channels='
            f'{self.out_channels}, memory={self.memory}'
        )
        if self.future_sight:
            s += f', future_sight={self.memory}'
        if self.kernel_width > 1:
            s += f', width={self.kernel_width}'
        if self.bias is not None:
            s += ', bias=True'
        s += ')'
        return s

    def forward(self, input):
        if self.future_sight:
            weight = self.weight
        else:
            weight = self.weight * self.mask
        return polyconv2d(
            input, weight=weight, bias=self.bias, padding=self.padding,
            include_const=self.include_const)
