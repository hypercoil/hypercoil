# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for specialised matrix operations
"""
import torch
import numpy as np
from scipy.linalg import toeplitz as toeplitz_ref
from hypernova.functional import (
    invert_spd, toeplitz
)


tol = 1e-6
testf = lambda out, ref: np.allclose(out, ref, atol=tol)


A = np.random.rand(10, 10)
A = A @ A.T
c = np.random.rand(3)
r = np.random.rand(3)
C = np.random.rand(3, 3)
R = np.random.rand(4, 3)
f = np.random.rand(3)
At = torch.Tensor(A)
ct = torch.Tensor(c)
rt = torch.Tensor(r)
Ct = torch.Tensor(C)
Rt = torch.Tensor(R)
ft = torch.Tensor(f)


def test_invert_spd():
    out = invert_spd(At).numpy()
    ref = np.linalg.inv(A)
    assert testf(out, ref)


def test_toeplitz():
    out = toeplitz(ct, rt).numpy()
    ref = toeplitz_ref(c, r)
    assert testf(out, ref)


def test_toeplitz_stack():
    out = toeplitz(Ct, Rt).numpy()
    ref = np.stack([toeplitz_ref(c, r) for c, r in zip(C.T, R.T)])
    assert testf(out, ref)


def test_toeplitz_extend():
    dim = (8, 8)
    out = toeplitz(Ct, Rt, dim=dim)
    Cx, Rx = np.zeros((dim[0], C.shape[1])), np.zeros((dim[1], R.shape[1]))
    Cx[:C.shape[0], :] = C
    Rx[:R.shape[0], :] = R
    ref = np.stack([toeplitz_ref(c, r) for c, r in zip(Cx.T, Rx.T)])
    assert testf(out, ref)


def test_toeplitz_fill():
    dim = (8, 8)
    out = toeplitz(Ct, Rt, dim=dim, fill_value=ft)
    Cx, Rx = (np.zeros((dim[0], C.shape[1])) + f,
              np.zeros((dim[1], R.shape[1])) + f)
    Cx[:C.shape[0], :] = C
    Rx[:R.shape[0], :] = R
    ref = np.stack([toeplitz_ref(c, r) for c, r in zip(Cx.T, Rx.T)])
    assert testf(out, ref)
