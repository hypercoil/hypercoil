# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data synthesis: pass band
~~~~~~~~~~~~~~~~~~~~~~~~~
Synthesise a dataset for the frequency-domain filtering autoencoder test.
"""
import numpy as np
from scipy.fft import rfft, irfft
from functools import reduce
from scipy.spatial.distance import squareform
from .synth_lgmix import create_mixture, observed_signals


DEFAULT_BANDS = [
    (0.05, 0.1),
    (0.1, 0.3),
    (0.3, 0.6)
]


APPROX_ORTHO_THRESH = 0.1


def synthesise(n=1000, d=7, p=100, bands=DEFAULT_BANDS):
    sources = np.random.randn(d, n)
    local = np.random.randn(p, n)
    mixtures = [create_mixture(d, p) for _ in bands]
    sources_filt = [bp_signals(sources, band, n) for band in bands]
    # Verify approximate orthogonality
    Z = np.stack(sources_filt)
    for i in range(len(bands)):
        cc = np.corrcoef(Z[:, i, :].squeeze())
        assert np.all(
            cc[np.triu_indices_from(cc, 1)] < APPROX_ORTHO_THRESH), (
                'Specified frequency bands are not approximately '
                'orthogonal'
            )
    # Fill unused bands with uncorrelated noise
    bandfill = bs_signals(local, bands, n)
    signal = collate_observed_signals(mixtures, sources_filt, local, bandfill)
    # Extract the true states (according to shared variance).
    statevar = [np.corrcoef(m) for m in mixtures]
    return signal, statevar, bands


def bp_signals(sources, band, n):
    lp, hp = band
    sources_fft = rfft(sources, n=n)
    n_bins = sources_fft.shape[-1]
    lp = int(np.floor(lp * n_bins))
    hp = int(np.ceil(hp * n_bins))
    sources_fft[:, :lp] = 0
    sources_fft[:, hp:] = 0
    sources_filt = irfft(sources_fft, n=n)
    return ((sources_filt.T - sources_filt.T.mean(0)) /
            sources_filt.T.std(0)).T


def bs_signals(sources, bands, n):
    sources_fft = rfft(sources, n=n)
    n_bins = sources_fft.shape[-1]
    for band in bands:
        lp, hp = band
        lp = int(np.floor(lp * n_bins))
        hp = int(np.ceil(hp * n_bins))
        print(lp, hp)
        sources_fft[:, lp:hp] = 0
    sources_filt = irfft(sources_fft, n=n)
    return ((sources_filt.T - sources_filt.T.mean(0)) /
            sources_filt.T.std(0)).T


def collate_observed_signals(mixtures, sources_filt, local, bandfill):
    observed = [
        observed_signals(m, s, local)
        for m, s in zip(mixtures, sources_filt)
    ]
    collate = lambda a, b: a + b
    signal = reduce(collate, observed) + bandfill
    return signal
