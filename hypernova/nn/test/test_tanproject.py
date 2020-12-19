# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for tangent/cone projection layer
"""
import torch
from hypernova.nn import TangentProject, BatchTangentProject
from hypernova.init.semidefinite import (
    SPDEuclideanMean,
    SPDHarmonicMean,
    SPDLogEuclideanMean,
    SPDGeometricMean
)


testf = torch.allclose


A = torch.rand(30, 10, 100)
A = A @ A.transpose(-2, -1)
Z = torch.rand(100, 10, 10)
Z = Z + Z.transpose(-1, -2)


def test_tangent_forward():
    tang = TangentProject(
        A, mean_specs=[
            SPDGeometricMean(psi=0.1),
            SPDEuclideanMean(),
            SPDEuclideanMean(),
            SPDLogEuclideanMean(),
            SPDHarmonicMean()],
        recondition=1e-5)
    Y = tang(A)
    tang(Y, 'cone')
    tang(Z, 'cone')


def test_batch_tangent_forward():
    btang = BatchTangentProject(
        mean_specs=[
            SPDGeometricMean(psi=0.1),
            SPDEuclideanMean(),
            SPDEuclideanMean(),
            SPDLogEuclideanMean(),
            SPDHarmonicMean()],
        recondition=1e-5)
    Y = btang(A)
    btang(Y, 'cone')
    btang(Z, 'cone')
