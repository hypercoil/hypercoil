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
