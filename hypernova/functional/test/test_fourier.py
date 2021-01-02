# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for Fourier-domain filtering
"""
import pytest
import torch
import numpy as np
from scipy.fft import rfft, irfft
from hypernova.functional import (
    product_filter
)


class TestFourier:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.tol = 1e-7
        self.approx = lambda out, ref: np.allclose(out, ref, atol=self.tol)

        self.N = 100
        self.X = np.random.rand(7, self.N)
        self.Xt = torch.Tensor(self.X)

    def scipy_product_filter(self, X, weight):
        return irfft(weight * rfft(X))

    def uniform_attenuator(self):
        return 0.5 * torch.ones(self.N // 2 + 1)

    def bandpass_filter(self):
        weight = torch.ones(self.N // 2 + 1)
        weight[:10] = 0
        weight[20:] = 0
        return weight

    def test_bandpass(self):
        wt = self.bandpass_filter()
        w = wt.numpy()
        out = product_filter(self.Xt, wt).numpy()
        ref = self.scipy_product_filter(self.X, w)
        assert self.approx(out, ref)

    def test_attenuation(self):
        wt = self.uniform_attenuator()
        out = product_filter(self.Xt, wt).numpy()
        ref = 0.5 * self.X
        assert self.approx(out, ref)
