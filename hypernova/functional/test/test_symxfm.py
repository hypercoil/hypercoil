# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for covariance, correlation, and derived measures
"""
import numpy as np
import torch
from scipy.linalg import expm, logm, sqrtm
from hypernova import (
    symexp, symlog, symsqrt
)


tol = 5e-7
testf = lambda out, ref: np.allclose(out, ref, atol=tol)


A = np.random.rand(10, 10)
A = A @ A.T
At = torch.Tensor(A)


def test_expm():
    out = symexp(At).numpy()
    ref = expm(A)
    testf(out, ref)


def test_logm():
    out = symlog(At).numpy()
    ref = logm(A)
    testf(out, ref)


def test_sqrtm():
    out = symsqrt(At).numpy()
    ref = sqrtm(A)
    testf(out, ref)
