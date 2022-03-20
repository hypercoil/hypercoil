# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for smoothness penalty
"""
import pytest
import torch
from hypercoil.loss import (
    SmoothnessPenalty
)


class TestSmoothnessPenalty:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.X = torch.Tensor([
            [0, 0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4, 0.5],
            [0.4, 0.5, 0.6, 0.7]
        ])
        self.y0 = torch.tensor(0.2828426957130432)
        self.y1 = torch.tensor(0.17320507764816284)

    def test_smoothness_ax0(self):
        reg = SmoothnessPenalty(axis=0)
        y_hat = reg(self.X)
        assert(torch.isclose(y_hat, self.y0))

    def test_smoothness_ax1(self):
        reg = SmoothnessPenalty(axis=1)
        y_hat = reg(self.X)
        assert(torch.isclose(y_hat, self.y1))
