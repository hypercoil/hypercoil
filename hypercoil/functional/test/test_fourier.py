# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for Fourier-domain filtering
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft
from scipy.signal import hilbert, chirp
from pkg_resources import resource_filename as pkgrf
from hypercoil.functional import (
    product_filter, unwrap, analytic_signal,
    envelope, instantaneous_frequency, env_inst
)
from hypercoil.functional.fourier import product_filtfilt


class TestFourier:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.tol = 1e-6
        self.approx = lambda out, ref: np.allclose(out, ref, atol=self.tol)

        self.N = 100
        self.X = np.random.rand(7, self.N)

        self.results = pkgrf(
            'hypercoil',
            'results/'
        )

    def scipy_product_filter(self, X, weight):
        return irfft(weight * rfft(X))

    def uniform_attenuator(self):
        return 0.5 * np.ones(self.N // 2 + 1)

    def bandpass_filter(self):
        weight = np.ones(self.N // 2 + 1)
        weight[:10] = 0
        weight[20:] = 0
        return weight

    def test_bandpass(self):
        w = self.bandpass_filter()
        out = product_filter(self.X, w)
        ref = self.scipy_product_filter(self.X, w)
        assert self.approx(out, ref)

    def test_zerophase_filter(self):
        w = (np.random.rand(self.N // 2 + 1) +
             1j * np.random.rand(self.N // 2 + 1))
        in_phase = jnp.angle(jnp.fft.rfft(self.X))
        out = product_filter(self.X, w)
        out_phase = jnp.angle(jnp.fft.rfft(out))
        assert not jnp.allclose(in_phase, out_phase, atol=1e-6)

        out = product_filtfilt(self.X, w)
        out_phase = jnp.angle(jnp.fft.rfft(out))
        assert jnp.allclose(in_phase, out_phase, atol=1e-6)

    def test_attenuation(self):
        w = self.uniform_attenuator()
        out = product_filter(self.X, w)
        ref = 0.5 * self.X
        assert self.approx(out, ref)

    def test_unwrap(self):
        # Replicate the numpy doc examples
        phase = np.linspace(0, np.pi, num=5)
        phase[3:] += np.pi
        ref = np.unwrap(phase)
        out = unwrap(phase)
        assert self.approx(ref, out)

        out = unwrap(np.array([0., 1, 2, -1, 0]), period=4)
        ref = np.array([0, 1, 2, 3, 4])
        assert self.approx(ref, out)

        ref = np.linspace(0, 720, 19) - 180
        phase = np.linspace(0, 720, 19) % 360 - 180
        out = unwrap(phase, period=360)
        assert self.approx(ref, out)

        phase = np.random.randint(
            20, size=(10, 10, 10)).astype(float)
        ref = np.unwrap(phase, axis=-2)
        out = unwrap(phase, axis=-2)
        assert np.max(np.abs(phase - out)) > 1e-5
        assert np.allclose(ref, out, atol=1e-5)

        junwrap = jax.jit(unwrap, static_argnames=('axis',))
        out2 = junwrap(phase, axis=-2)
        assert np.allclose(out2, out)

    def test_hilbert_transform(self):
        out = analytic_signal(self.X)
        ref = hilbert(self.X)
        assert self.approx(out, ref)
        assert np.allclose(self.X, out.real, atol=1e-6)

        X = np.random.randn(3, 10, 50, 5)
        ref = hilbert(X, axis=-2)
        gradient = jax.grad(lambda x: jnp.angle(analytic_signal(x)).sum())
        out = analytic_signal(X, -2)
        assert np.allclose(out, ref, atol=1e-6)
        assert np.allclose(X, out.real, atol=1e-6)
        X_grad = gradient(X)
        assert X_grad is not None

    def test_hilbert_envelope(self):
        # replicating the example from the scipy documentation
        duration = 1.0
        fs = 400.0
        samples = int(fs*duration)
        t = np.arange(samples) / fs

        signal = chirp(t, 20.0, t[-1], 100.0)
        signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )

        amplitude_envelope = envelope(signal)
        inst_freq = instantaneous_frequency(signal, fs=400)

        fig, (ax0, ax1) = plt.subplots(nrows=2)

        ax0.plot(t, signal, label='signal')
        ax0.plot(t, amplitude_envelope, label='envelope')
        ax0.set_xlabel("time in seconds")
        ax0.legend()

        ax1.plot(t[1:], inst_freq)
        ax1.set_xlabel("time in seconds")
        ax1.set_ylim(0.0, 120.0)
        fig.tight_layout()

        fig.savefig(f'{self.results}/hilbert_separate.png')

        amplitude_envelope, inst_freq, _ = env_inst(signal, fs=400)

        fig, (ax0, ax1) = plt.subplots(nrows=2)

        ax0.plot(t, signal, label='signal')
        ax0.plot(t, amplitude_envelope, label='envelope')
        ax0.set_xlabel("time in seconds")
        ax0.legend()

        ax1.plot(t[1:], inst_freq)
        ax1.set_xlabel("time in seconds")
        ax1.set_ylim(0.0, 120.0)
        fig.tight_layout()

        fig.savefig(f'{self.results}/hilbert_onecall.png')
