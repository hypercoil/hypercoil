# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Filter synthesis
~~~~~~~~~~~~~~~~
Synthesise some simple ground truth datasets for testing filter learning.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from functools import reduce
from scipy.fft import rfft, irfft
from .mix import (
    synthesise_mixture,
    mix_data
)
from hypercoil.functional import complex_decompose


APPROX_ORTHO_THRESH = 0.1


def synthesise_across_bands(
    bands,
    time_dim=1000,
    observed_dim=7,
    latent_dim=100,
    seed=None
):
    """
    Synthesise a dataset that has a different correlation structure in
    different frequency bands.

    Parameters
    ----------
    bands : list(tuple)
        Frequency bands into which structured, correlated data is injected.
        The data in each band will have a different structure. Denote as
        (high pass, low pass) as a fraction of Nyquist.
    time_dim : int (default 1000)
        Number of time points per signal.
    observed_dim : int (default 7)
        Number of observed signals to return.
    latent_dim : int (default 100)
        Number of latent signals to synthesise.
    seed : int (default None)
        Seed for RNG.
    """
    np.random.seed(seed)
    sources = np.random.randn(observed_dim, time_dim)
    local = np.random.randn(observed_dim, time_dim)
    mix_seed = [None, None, None]
    if seed is not None: mix_seed = list(range(seed, seed + len(bands)))
    mixtures = [
        synthesise_mixture(
            time_dim=time_dim,
            observed_dim=observed_dim,
            latent_dim=latent_dim,
            subject_dim=1,
            include_local=True,
            local_scale=0.25,
            lp=band[1],
            hp=band[0],
            return_mix_matrix=True,
            seed=mix_seed[i]
        )
        for i, band in enumerate(bands)
    ]
    sources_filt, mixtures = list(zip(*mixtures))
    #sources_filt = [bp_signals(sources, band, time_dim) for band in bands]
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
    bandfill = bs_signals(local, bands, time_dim)
    signal = collate_observed_signals(sources_filt, local, bandfill)
    # Extract the true states (according to shared variance).
    statevar = [np.corrcoef(m) for m in mixtures]
    return signal, statevar, bands


def bs_signals(sources, bands, n):
    """
    Fill frequency bands outside the specified list using the specified
    sources.
    """
    sources_fft = rfft(sources, n=n)
    n_bins = sources_fft.shape[-1]
    for band in bands:
        hp, lp = band
        hp = int(np.floor(hp * n_bins))
        lp = int(np.ceil(lp * n_bins))
        sources_fft[:, hp:lp] = 0
    sources_filt = irfft(sources_fft, n=n)
    return ((sources_filt.T - sources_filt.T.mean(0)) /
            sources_filt.T.std(0)).T


def collate_observed_signals(sources_filt, local, bandfill):
    collate = lambda a, b: a + b
    signal = reduce(collate, sources_filt) + bandfill
    return signal


def plot_frequency_partition(bands, filter, save=None):
    """
    Plotting utility when learning a partition over frequencies.
    """
    freq_dim = filter.weight.shape[-1]
    plt.figure(figsize=(12, 8))
    for (hp, lp) in bands:
        plt.axvline(hp, ls=':', color='grey')
        plt.axvline(lp, ls=':', color='grey')
    # Omit the last weight. Here we assume it corresponds to a rejection band.
    ampl = complex_decompose(filter.weight)[0][:-1]
    for s in ampl:
        plt.plot(np.linspace(0, 1, freq_dim), s.detach().numpy())
        plt.ylim([0, 1])
        plt.xlim([0, 1])
    if save:
        plt.savefig(save, bbox_inches='tight')


def plot_mvkurtosis(fftfilter, weight, input, bands, nu=1, l2=0.01, save=None):
    from hypercoil.loss import MultivariateKurtosis
    freq_dim = fftfilter.dim
    freq = np.linspace(0, 1, freq_dim)
    out = np.zeros(freq_dim)
    mvk = MultivariateKurtosis(nu=nu, l2=l2)
    for hp, lp in bands:
        with torch.no_grad():
            fftfilter.preweight.zero_()
            hp = int(freq_dim * hp)
            lp = int(freq_dim * lp)
            fftfilter.preweight[:, hp:lp] = 1.
        out[hp:lp] = -mvk(fftfilter(input)).detach()
    plt.figure(figsize=(12, 8))
    plt.plot(freq, (out / out.max()), c='#999999')
    plt.plot(freq, weight.squeeze().detach(), c='#7722CC')
    plt.legend(['MV Kurtosis', 'Transfer'])
    plt.ylim((0, 1))
    plt.xlim((0, 1))
    plt.xlabel('Frequency')
    if save:
        plt.savefig(save, bbox_inches='tight')
