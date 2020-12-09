# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Delta-plus initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~
Initialise parameters as a set of delta functions, plus Gaussian noise.
"""
import torch


def deltaplus_init_(tensor, loc=None, scale=None, std=0.2):
    loc = loc or tuple([x // 2 for x in tensor.size()])
    scale = scale or 1
    val = torch.zeros_like(tensor)
    val[loc] += scale
    val += torch.randn(tensor.size()) * std
    val.type(tensor.dtype)
    tensor[:] = val
