# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for polynomial convolution
"""
import torch
from hypernova.fourier import (
    productfilter
)


testf = torch.allclose


N = 100
X = torch.rand(7, N)


def bandpass_filter():
    weight = torch.ones(N // 2 + 1)
    weight[:10] = 0
    weight[20:] = 0
    return weight
