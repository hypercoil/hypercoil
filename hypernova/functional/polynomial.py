# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Polynomial convolution
~~~~~~~~~~~~~~~~~~~~~~
Functions supporting polynomial convolution of time series.
"""
import torch


def polychan(X, degree=2, include_const=False):
    if X.dim() > 2:
        pass
    elif X.dim() == 2:
        X = X.view(1, *X.size())
    elif X.dim() == 1:
        X = X.view(1, -1)
    stack = [X]
    for _ in range(degree - 1):
        stack += [stack[-1] * X]
    if include_const:
        stack = [torch.ones_like(X)] + stack
    return torch.stack(stack, 1)


def polyconv(X, weight, include_const=False, bias=None):
    degree = weight.size(1) - include_const
    padding = (0, weight.size(-1) // 2)
    X = polychan(X, degree=degree, include_const=include_const)
    return torch.conv2d(X, weight, bias=bias, stride=1, padding=padding)
