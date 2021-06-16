# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for symmetric bimodal penalty
"""
import pytest
import torch
from hypercoil.reg import (
    SymmetricBimodal
)


class TestSymmetricBimodal:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.X1 = torch.Tensor([
            [0.2, 0, 1, 0.7, 1],
            [1.2, 0, 0.8, -0.2, 0],
            [0, 1, 0.3, 0, 1]
        ])
        self.X2 = torch.Tensor(
            [0.8, 0.5, 0.1]
        )
        self.y1 = torch.tensor(0.58309518948453)
        self.y2 = torch.tensor(0.65)

    def test_symbm_l2(self):
        reg = SymmetricBimodal()
        y_hat = reg(self.X1)
        assert(torch.isclose(self.y1, y_hat))

    def test_symbm_l1(self):
        reg = SymmetricBimodal(norm=1, modes=(0.95, 0.05))
        y_hat = reg(self.X2)
        assert(torch.isclose(self.y2, y_hat))
