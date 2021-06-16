# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for tangent/cone projection layer
"""
import pytest
import torch
from hypercoil.nn import TangentProject, BatchTangentProject
from hypercoil.init.semidefinite import (
    SPDEuclideanMean,
    SPDHarmonicMean,
    SPDLogEuclideanMean,
    SPDGeometricMean
)


class TestTanProject:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        A = torch.rand(30, 10, 100)
        self.A = A @ A.transpose(-2, -1)
        Z = torch.rand(100, 10, 10)
        self.Z = Z + Z.transpose(-1, -2)

        self.approx = torch.allclose

    def test_tangent_cone_forward(self):
        tang = TangentProject(
            self.A, mean_specs=[
                SPDGeometricMean(psi=0.1),
                SPDEuclideanMean(),
                SPDEuclideanMean(),
                SPDLogEuclideanMean(),
                SPDHarmonicMean()],
            recondition=1e-5)
        Y = tang(self.A)
        tang(Y, 'cone')
        tang(self.Z, 'cone')

    def test_batch_tangent_cone_forward(self):
        btang = BatchTangentProject(
            mean_specs=[
                SPDGeometricMean(psi=0.1),
                SPDEuclideanMean(),
                SPDEuclideanMean(),
                SPDLogEuclideanMean(),
                SPDHarmonicMean()],
            recondition=1e-5)
        Y = btang(self.A)
        btang(Y, 'cone')
        btang(self.Z, 'cone')
