# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for differentiable terminals.
"""
import pytest
import torch
import matplotlib.pyplot as plt
from pkg_resources import resource_filename as pkgrf
from hypercoil.functional import hybrid_interpolate


class TestInterpolation:

    def synthesise_data(self, seed, mode):
        torch.manual_seed(seed)
        k = 2000

        fsp = torch.zeros((k // 2) + 1)
        fsp[3] = 1
        fsp[15] = 1
        fsp[22] = 1
        #fsp[222] = 1
        t = torch.fft.irfft(fsp)
        t /= t.std()
        noise = torch.randn(3, k)
        if mode == 'interpolate':
            mask0 = (torch.rand(3, k) > 0.6)
            mask1 = ((torch.arange(k) <= (k // 4)) +
                     (torch.arange(k) >= (3 * k // 4)))
        elif mode == 'extrapolate':
            mask0 = (torch.rand(3, k) > 0.3)
            mask1 = (torch.arange(k) <= (k // 4))
        mask = mask0 * mask1
        seen = torch.where(mask, t, noise).unsqueeze(-2).squeeze()

        seen = seen - seen.mean()
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
        ax1.plot(t.t(), color='blue')
        ax1.plot(rec.squeeze()[0].t(), color='red')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.legend(['Observed', 'Actual', 'Reconstructed'])

        ax2.plot(torch.fft.rfft(seen[0]), color='grey')
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.plot(torch.fft.rfft(t), color='blue')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4.plot(torch.fft.rfft(rec.squeeze()[0]), color='red')
        ax4.set_xticks([])
        ax4.set_yticks([])

        ax5.plot(torch.fft.rfft(seen[0])[:50], color='grey')
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax6.plot(torch.fft.rfft(t)[:50], color='blue')
        ax6.set_xticks([])
        ax6.set_yticks([])
        ax7.plot(torch.fft.rfft(rec.squeeze()[0])[:50], color='red')
        ax7.set_xticks([])
        ax7.set_yticks([])

        results = pkgrf(
            'hypercoil',
            'results/'
        )
        fig.savefig(f'{results}/hybrid_{path}.png', bbox_inches='tight')

    def test_hybrid_interpolate(self):
        seen, t, mask = self.synthesise_data(77, 'interpolate')

        rec = hybrid_interpolate(
            seen.view(3, 1, 1, -1),
            mask.view(3, 1, 1, -1),
            max_weighted_stage=5,
            frequency_thresh=0.8
        )

        self.plot_figure(rec, seen, t, 'interpolate')


    def test_hybrid_extrapolate(self):
        seen, t, mask = self.synthesise_data(77, 'extrapolate')

        rec = hybrid_interpolate(
            seen.view(3, 1, 1, -1),
            mask.view(3, 1, 1, -1),
            max_weighted_stage=5,
            frequency_thresh=0.9
        )

        self.plot_figure(rec, seen, t, 'extrapolate')
