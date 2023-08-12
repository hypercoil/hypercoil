# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for centre of mass
"""
import pytest
import numpy as np
from hypercoil.functional.cmass import cmass, cmass_coor


#TODO: Unit tests still needed for
# - "centres of mass" in spherical coordinates
# - regularisers: `cmass_reference_displacement` and `diffuse`


class TestCentreOfMass:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.X = np.array([
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1]
        ])
        self.X_all = np.array([1.5, 1.5])
        self.X_0 = np.array([1, 2, 1.5, 1.5])
        self.X_1 = np.array([1.5, 1.5, 1, 2])
        self.Y = np.random.rand(5, 3, 4, 4)
        self.Y = (self.Y > 0.5).astype(float)

        coor = np.arange(4).reshape(1, 4)
        self.Xcoor = np.stack([
            np.tile(coor, (4, 1)),
            np.tile(coor.reshape(4, 1), (1, 4))
        ])
        self.Ycoor = np.stack([
            np.broadcast_to(np.arange(5).reshape(-1, 1, 1, 1), self.Y.shape),
            np.broadcast_to(np.arange(3).reshape(1, -1, 1, 1), self.Y.shape),
            np.broadcast_to(np.arange(4).reshape(1, 1, -1, 1), self.Y.shape),
            np.broadcast_to(np.arange(4).reshape(1, 1, 1, -1), self.Y.shape),
        ])

    def test_cmass_negatives(self):
        out = cmass(self.X, [0])
        ref = cmass(self.X, [-2])
        assert np.all(out == ref)
        out = cmass(self.Y, [-1, -3])
        ref = cmass(self.Y, [3, 1])
        assert np.allclose(out, ref)
        out = cmass(self.Y, [0, -1])
        ref = cmass(self.Y, [0, 3])
        assert np.allclose(out, ref)

    def test_cmass_values(self):
        out = cmass(self.X)
        ref = self.X_all
        assert np.allclose(out, ref)
        out = cmass(self.X, [0]).squeeze()
        ref = self.X_0
        assert np.allclose(out, ref)
        out = cmass(self.X, [1]).squeeze()
        ref = self.X_1
        assert np.allclose(out, ref)

    def test_cmass_dim(self):
        out = cmass(self.Y, [-1, -3])
        assert out.shape == (5, 4, 2)
        out = cmass(self.Y, [-2])
        assert out.shape == (5, 3, 4, 1)
        out = cmass(self.Y, [0, -3, -2])
        assert out.shape == (4, 3)

    def test_cmass_coor(self):
        out = cmass_coor(self.X.reshape(1, -1), self.Xcoor.reshape(2, -1))
        ref = cmass(self.X)
        assert np.allclose(out, ref)

        out = cmass_coor(self.Y.reshape(1, -1), self.Ycoor.reshape(4, -1))
        ref = cmass(self.Y)
        assert np.allclose(out.squeeze(), ref.squeeze())

    def test_cmass_equivalence(self):
        out = cmass_coor(self.X.reshape(1, -1), self.Xcoor.reshape(2, -1))
        ref = cmass(self.X)
        assert np.allclose(out, ref)

        out = cmass_coor(self.Y.reshape(1, -1), self.Ycoor.reshape(4, -1))
        ref = cmass(self.Y)
        assert np.allclose(out.squeeze(), ref.squeeze())
