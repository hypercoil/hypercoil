# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for polynomial convolution
"""
import pytest
import torch
from hypernova.functional import (
    polyconv2d
)


def known_filter():
    weight = torch.Tensor([
        [0, 0, 1, 0, 0],
        [0, 0, 0.3, 0, 0],
        [0, 0, -0.1, 0, 0]
    ])
    return weight.view(1, weight.size(0), 1, weight.size(1))


class TestPolynomial:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.X = torch.rand(7, 100)
        self.approx = torch.allclose

    def test_polyconv2d(self):
        out = polyconv2d(self.X, known_filter())
        ref = self.X + 0.3 * self.X ** 2 - 0.1 * self.X ** 3
        assert self.approx(out, ref)
