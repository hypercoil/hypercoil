# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for base loss modules and schemes
"""
import pytest
import math
import torch
from hypercoil.loss import (
    LossApply,
    NormedLoss,
    ReducingLoss,
    LossScheme
)
from hypercoil.loss.base import wmean


class TestLossBase:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.A = torch.tensor([2, 0, 2, 0], dtype=torch.float)
        self.B = torch.tensor([1, 1, 1, 1], dtype=torch.float)
        self.C = torch.tensor([
            [2, 0, 2, 0],
            [1, 1, 1, 1]
        ], dtype=torch.float)

    def test(self):
        # The outer `apply` switches the order of the inputs, and then the
        # first input is passed to L2 and the second to L1.
        scheme = LossScheme([
            LossApply(
                NormedLoss(nu=1),
                apply=lambda xy: xy[0]),
            LossApply(
                NormedLoss(nu=1, p=1),
                apply=lambda xy: xy[1])],
            apply=lambda x, y: (y, x))

        out = scheme(x=self.A, y=self.B)
        assert out == 6

        out = scheme(x=self.B, y=self.A)
        assert out == 4 + math.sqrt(8)

    def test_wmean(self):
        z = torch.tensor([[
            [1., 4., 2.],
            [0., 9., 1.],
            [4., 6., 7.]],[
            [0., 9., 1.],
            [4., 6., 7.],
            [1., 4., 2.]
        ]])
        w = torch.ones_like(z)
        assert wmean(z, w) == torch.mean(z)
        w = torch.tensor([1., 0., 1.])
        assert torch.all(wmean(z, w, dim=1) == torch.tensor([
            [(1 + 4) / 2, (4 + 6) / 2, (2 + 7) / 2],
            [(0 + 1) / 2, (9 + 4) / 2, (1 + 2) / 2]
        ]))
        assert torch.all(wmean(z, w, dim=2) == torch.tensor([
            [(1 + 2) / 2, (0 + 1) / 2, (4 + 7) / 2],
            [(0 + 1) / 2, (4 + 7) / 2, (1 + 2) / 2]
        ]))
        w = torch.tensor([
            [1., 0., 1.],
            [0., 1., 1.]
        ])
        assert torch.all(wmean(z, w, dim=(0, 1)) == torch.tensor([
            [(1 + 4 + 4 + 1) / 4, (4 + 6 + 6 + 4) / 4, (2 + 7 + 7 + 2) / 4]
        ]))
        assert torch.all(wmean(z, w, dim=(0, 2)) == torch.tensor([
            [(1 + 2 + 9 + 1) / 4, (0 + 1 + 6 + 7) / 4, (4 + 7 + 4 + 2) / 4]
        ]))
