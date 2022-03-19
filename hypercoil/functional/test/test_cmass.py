# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for centre of mass
"""
import pytest
import torch
from hypercoil.functional.cmass import cmass, cmass_coor


#TODO: Unit tests still needed for
# - "centres of mass" in spherical coordinates
# - regularisers: `cmass_reference_displacement` and `diffuse`


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

        coor = torch.arange(4).view(1, 4)
        self.Xcoor = torch.stack([
            coor.tile(4, 1),
            coor.view(4, 1).tile(1, 4)
        ])
        self.Ycoor = torch.stack([
            torch.arange(5).view(-1, 1, 1, 1).broadcast_to(self.Y.shape),
            torch.arange(3).view(1, -1, 1, 1).broadcast_to(self.Y.shape),
            torch.arange(4).view(1, 1, -1, 1).broadcast_to(self.Y.shape),
            torch.arange(4).view(1, 1, 1, -1).broadcast_to(self.Y.shape),
        ])

        if torch.cuda.is_available():
            self.XC = self.X.clone().cuda()
            self.YC = self.Y.clone().cuda()
            self.XcoorC = self.Xcoor.clone().cuda()
            self.YcoorC = self.Ycoor.clone().cuda()

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

    def test_cmass_coor(self):
        out = cmass_coor(self.X.view(1, -1), self.Xcoor.view(2, -1))
        ref = cmass(self.X)
        assert torch.allclose(out, ref)

        out = cmass_coor(self.Y.view(1, -1), self.Ycoor.view(4, -1))
        ref = cmass(self.Y)
        assert torch.allclose(out.squeeze(), ref.squeeze())

    @pytest.mark.cuda
    def test_cmass_equivalence_cuda(self):
        out = cmass_coor(self.XC.view(1, -1), self.XcoorC.view(2, -1))
        ref = cmass(self.XC)
        assert torch.allclose(out, ref)

        out = cmass_coor(self.YC.view(1, -1), self.YcoorC.view(4, -1))
        ref = cmass(self.YC)
        assert torch.allclose(out.squeeze(), ref.squeeze())
