# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for interpolation functions for evenly sampled time series.
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pkg_resources import resource_filename as pkgrf
from hypercoil.functional.interpolate import (
    hybrid_interpolate, spectral_interpolate,
    linear_interpolate, weighted_interpolate,
    centred_square_kernel, _weighted_interpolate_stage
)


class TestInterpolation:

    def synthesise_data(self, key, mode, mask_key=None):
        if mask_key is None:
            mask_key = key
        key = jax.random.PRNGKey(key)
        mask_key = jax.random.PRNGKey(mask_key)

        k = 2000

        fsp = np.zeros((k // 2) + 1)
        fsp[3] = 1
        fsp[15] = 1
        fsp[22] = 1
        #fsp[222] = 1
        t = np.fft.irfft(fsp)
        t -= t.mean(-1, keepdims=True)
        t /= t.std()
        noise = jax.random.normal(key, (3, k))
        noise -= noise.mean(-1, keepdims=True)
        if mode == 'interpolate':
            mask0 = jax.random.bernoulli(mask_key, 0.4, (3, k))
            mask1 = ((jnp.arange(k) <= (k // 4)) +
                     (jnp.arange(k) >= (3 * k // 4)))
        elif mode == 'extrapolate':
            mask0 = jax.random.bernoulli(mask_key, 0.7, (3, k))
            mask1 = (jnp.arange(k) <= (k // 4))
        mask = mask0 * mask1
        seen = jnp.where(mask, t, noise)

        return seen, t, mask

    def plot_figure(self, rec, seen, t, path):
        fig = plt.figure(figsize=(12, 12))

        gs = fig.add_gridspec(3, 3)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[1, 2])
        ax5 = fig.add_subplot(gs[2, 0])
        ax6 = fig.add_subplot(gs[2, 1])
        ax7 = fig.add_subplot(gs[2, 2])

        ax1.plot(seen[0], color='grey')
        ax1.plot(t.T, color='blue')
        ax1.plot(rec[0, 0, 0, :], color='red')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.legend(['Observed', 'Actual', 'Reconstructed'])

        ax2.plot(np.fft.rfft(seen[0]), color='grey')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.plot(np.fft.rfft(t), color='blue')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4.plot(np.fft.rfft(rec[0, 0, 0, :]), color='red')
        ax4.set_xticks([])
        ax4.set_yticks([])

        ax5.plot(np.fft.rfft(seen[0])[:50], color='grey')
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax6.plot(np.fft.rfft(t)[:50], color='blue')
        ax6.set_xticks([])
        ax6.set_yticks([])
        ax7.plot(np.fft.rfft(rec[0, 0, 0, :])[:50], color='red')
        ax7.set_xticks([])
        ax7.set_yticks([])

        results = pkgrf(
            'hypercoil',
            'results/'
        )
        fig.savefig(f'{results}/interpolate_{path}.png', bbox_inches='tight')

    def test_hybrid_interpolate(self):
        seen0, t, mask = self.synthesise_data(77, 'interpolate', mask_key=77)
        seen1, _, _ = self.synthesise_data(18, 'interpolate', mask_key=77)

        seen = np.concatenate(
            (seen0.reshape(3, 1, 1, -1), seen1.reshape(3, 1, 1, -1)),
            axis=-2
        )

        rec = hybrid_interpolate(
            seen,
            mask.reshape(3, 1, 1, -1),
            max_consecutive_linear=15,
            frequency_thresh=0.8
        )

        self.plot_figure(
            np.array(rec)[[-1]][:, [0]][..., [-1], :],
            seen[[-1], 0, -1, :],
            t,
            'hybrid-interpolate'
        )

    def test_hybrid_extrapolate(self):
        seen, t, mask = self.synthesise_data(77, 'extrapolate')

        rec = hybrid_interpolate(
            seen.reshape(3, 1, 1, -1),
            mask.reshape(3, 1, 1, -1),
            max_consecutive_linear=15,
            frequency_thresh=0.9
        )

        self.plot_figure(rec, seen, t, 'hybrid-extrapolate')

    def test_linear_interpolate(self):
        seen, t, mask = self.synthesise_data(77, 'interpolate')

        rec = linear_interpolate(
            seen.reshape(3, 1, 1, -1),
            mask.reshape(3, 1, 1, -1)
        )

        self.plot_figure(rec, seen, t, 'linear-interpolate')

    def test_linear_extrapolate(self):
        seen, t, mask = self.synthesise_data(77, 'extrapolate')

        rec = linear_interpolate(
            seen.reshape(3, 1, 1, -1),
            mask.reshape(3, 1, 1, -1)
        )

        self.plot_figure(rec, seen, t, 'linear-extrapolate')

    def test_spectral_interpolate(self):
        seen, t, mask = self.synthesise_data(77, 'interpolate')

        rec = spectral_interpolate(
            seen.reshape(3, 1, 1, -1),
            mask.reshape(3, 1, 1, -1),
            thresh=0.8
        )

        self.plot_figure(rec, seen, t, 'spectral-interpolate')

    def test_spectral_extrapolate(self):
        seen, t, mask = self.synthesise_data(77, 'extrapolate')

        rec = spectral_interpolate(
            seen.reshape(3, 1, 1, -1),
            mask.reshape(3, 1, 1, -1),
            thresh=0.8
        )

        self.plot_figure(rec, seen, t, 'spectral-extrapolate')

    def test_weighted_interpolate(self):
        seen, t, mask = self.synthesise_data(77, 'interpolate')

        rec = weighted_interpolate(
            seen.reshape(3, 1, 1, -1),
            mask.reshape(3, 1, 1, -1),
            stages=list(range(10, 100)) + [1500]
        )

        self.plot_figure(rec, seen, t, 'weighted-interpolate')

    def test_weighted_interpolate_single_stage(self):
        seen, t, mask = self.synthesise_data(77, 'interpolate')

        kernel = centred_square_kernel(2, 1000)
        (rec, mask), _ = _weighted_interpolate_stage(seen, mask, kernel)
        rec = np.where(mask, rec, float('nan'))

        self.plot_figure(rec, seen, t, 'weighted-singlestage2-interpolate')

    def test_default_kernel(self):
        max_stage = 5
        ker = centred_square_kernel(3, max_stage)
        assert np.all(ker == np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]))
        ker = centred_square_kernel(2, max_stage)
        assert np.all(ker == np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]))
