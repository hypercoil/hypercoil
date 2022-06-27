# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for Fourier-domain filtering
"""
import pytest
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft
from scipy.signal import hilbert, chirp
from hypercoil.functional import (
    product_filter, unwrap, analytic_signal,
    envelope, instantaneous_frequency
)


#TODO: Unit tests missing for:
# - zero-phase filter
# - verify that our zero-phase filter is really zero-phase


class TestFourier:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.tol = 1e-7
        self.approx = lambda out, ref: np.allclose(out, ref, atol=self.tol)

        self.N = 100
        self.X = np.random.rand(7, self.N)
        self.Xt = torch.Tensor(self.X)

        if torch.cuda.is_available():
            self.XtC = self.Xt.clone().cuda()

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

    @pytest.mark.cuda
    def test_bandpass_cuda(self):
        wt = self.bandpass_filter()
        w = wt.numpy()
        out = product_filter(self.XtC, wt.cuda()).cpu().numpy()
        ref = self.scipy_product_filter(self.X, w)
        assert self.approx(out, ref)

    def test_unwrap(self):
        # Replicate the numpy doc examples
        phase = np.linspace(0, np.pi, num=5)
        phase[3:] += np.pi
        ref = np.unwrap(phase)
        out = unwrap(torch.tensor(phase))
        assert self.approx(ref, out)

        out = unwrap(torch.tensor([0., 1, 2, -1, 0]), period=4)
        ref = torch.tensor([0, 1, 2, 3, 4])
        assert self.approx(ref, out)

        ref = torch.linspace(0, 720, 19) - 180
        phase = torch.linspace(0, 720, 19) % 360 - 180
        out = unwrap(phase, period=360)
        assert self.approx(ref, out)

        phase = torch.randint(20, (10, 10, 10), dtype=torch.float)
        ref = np.unwrap(phase, axis=-2)
        out = unwrap(phase, axis=-2)
        assert (phase - out).abs().max() > 1e-5
        assert np.allclose(ref, out, atol=1e-5)

    def test_hilbert_transform(self):
        out = analytic_signal(self.Xt)
        ref = hilbert(self.X)
        assert self.approx(out, ref)
        assert np.allclose(self.Xt, out.real, atol=1e-6)

        X = torch.randn(3, 10, 50, 5)
        ref = hilbert(X, axis=-2)
        X.requires_grad = True
        out = analytic_signal(X, -2)
        assert np.allclose(out.detach(), ref, atol=1e-6)
        assert np.allclose(X.detach(), out.real.detach(), atol=1e-6)
        assert X.grad is None
        out.imag.sum().backward()
        assert X.grad is not None
