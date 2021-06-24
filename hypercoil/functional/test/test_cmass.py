# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for centre of mass
"""
import pytest
import torch
from hypercoil.functional.cmass import cmass


class TestCentreOfMass:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.X = torch.Tensor([
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ])
        self.X_all = torch.tensor([1.5, 1.5])
        self.X_0 = torch.tensor([1, 2, 1.5, 1.5])
        self.X_1 = torch.tensor([1.5, 1.5, 1, 2])
        self.Y = torch.rand(5, 3, 4, 4)
        self.Y = (self.Y > 0.5).float()

    def test_cmass_negatives(self):
        out = cmass(self.X, [0])
        ref = cmass(self.X, [-2])
        assert torch.all(out == ref)
        out = cmass(self.Y, [-1, -3])
        ref = cmass(self.Y, [3, 1])
        assert torch.allclose(out, ref)
        out = cmass(self.Y, [0, -1])
        ref = cmass(self.Y, [0, 3])
        assert torch.allclose(out, ref)

    def test_cmass_values(self):
        out = cmass(self.X)
        ref = self.X_all
        assert torch.allclose(out, ref)
        out = cmass(self.X, [0]).squeeze()
        ref = self.X_0
        assert torch.allclose(out, ref)
        out = cmass(self.X, [1]).squeeze()
        ref = self.X_1
        assert torch.allclose(out, ref)

    def test_cmass_dim(self):
        out = cmass(self.Y, [-1, -3])
        assert out.size() == torch.Size([5, 4, 2])
        out = cmass(self.Y, [-2])
        assert out.size() == torch.Size([5, 3, 4, 1])
        out = cmass(self.Y, [0, -3, -2])
        assert out.size() == torch.Size([4, 3])
