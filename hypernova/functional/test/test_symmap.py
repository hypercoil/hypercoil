# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for symmetric matrix maps
"""
import numpy as np
import torch
from scipy.linalg import expm, logm, sqrtm, sinm, funm
from hypernova.functional import (
    symmap, symexp, symlog, symsqrt
)


tol = 5e-7
rtol = 1e-4
testf = lambda out, ref: np.allclose(out, ref, atol=tol, rtol=rtol)


A = np.random.rand(10, 10)
AM = np.random.rand(200, 10, 10)
AS = np.random.rand(200, 10, 4)
A = A @ A.T
AM = AM @ np.swapaxes(AM, -1, -2)
AS = AS @ np.swapaxes(AS, -1, -2)
At = torch.Tensor(A)
AMt = torch.Tensor(AM)
ASt = torch.Tensor(AS)


def test_expm():
    out = symexp(At).numpy()
    ref = expm(A)
    assert testf(out, ref)


def test_logm():
    out = symlog(At).numpy()
    ref = logm(A)
    # Note that this is a very weak condition! This would likely
    # experience major improvement if pytorch develops a proper
    # logm function.
    assert np.allclose(out, ref, atol=1e-3, rtol=1e-3)


def test_sqrtm():
    out = symsqrt(At).numpy()
    ref = sqrtm(A)
    assert testf(out, ref)


def test_map():
    out = symmap(At, torch.sin).numpy()
    ref = funm(A, np.sin)
    assert testf(out, ref)
    ref = sinm(A)
    assert testf(out, ref)


def test_map_multidim():
    out = symmap(AMt, torch.exp).numpy()
    ref = np.stack([expm(AMi) for AMi in AM])
    assert testf(out, ref)


def test_singular():
    out = symmap(ASt, torch.log).numpy()
    assert np.all(np.logical_or(np.isnan(out), np.isinf(out)))
    out = symmap(ASt, torch.log, psi=1e-5).numpy()
    assert np.all(np.logical_not(np.logical_or(np.isnan(out), np.isinf(out))))
