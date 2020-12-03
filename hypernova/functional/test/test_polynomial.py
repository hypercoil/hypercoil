# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for polynomial convolution
"""
import torch
from hypernova.functional import (
    polyconv2d
)


testf = torch.allclose


X = torch.rand(7, 100)


def known_filter():
    weight = torch.Tensor([
        [0, 0, 1, 0, 0],
        [0, 0, 0.3, 0, 0],
        [0, 0, -0.1, 0, 0]
    ])
    return weight.view(1, weight.size(0), 1, weight.size(1))


def test_polyconv2d():
    out = polyconv2d(X, known_filter())
    ref = X + 0.3 * X ** 2 - 0.1 * X ** 3
    assert testf(out, ref)
