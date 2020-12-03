# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for Fourier-domain filtering
"""
import torch
import numpy as np
from scipy.fft import rfft, irfft
from hypernova.functional import (
    product_filter
)


tol = 1e-7
testf = lambda out, ref: np.allclose(out, ref, atol=tol)


N = 100
X = np.random.rand(7, N)
Xt = torch.Tensor(X)


def scipy_product_filter(X, weight):
    return irfft(weight * rfft(X))


def uniform_attenuator():
    return 0.5 * torch.ones(N // 2 + 1)


def bandpass_filter():
    weight = torch.ones(N // 2 + 1)
    weight[:10] = 0
    weight[20:] = 0
    return weight


def test_bandpass():
    wt = bandpass_filter()
    w = wt.numpy()
    out = product_filter(Xt, wt).numpy()
    ref = scipy_product_filter(X, w)
    assert testf(out, ref)


def test_attenuation():
    wt = uniform_attenuator()
    out = product_filter(Xt, wt).numpy()
    ref = 0.5 * X
    assert testf(out, ref)
