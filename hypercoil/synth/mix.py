# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Signal mixture synthesis
~~~~~~~~~~~~~~~~~~~~~~~~
Data synthesis using a linear signal mixture.
"""
import numpy as np
from scipy.fft import rfft, irfft
from scipy.stats import poisson


def synth_slow_signals(
        signal_dim=100,
        time_dim=200,
        subject_dim=1,
        lp=0.3,
        seed=None
    ):
    np.random.seed(seed)
    sources = np.random.rand(subject_dim, signal_dim, time_dim)
    sources_freq = rfft(sources, n=time_dim)
    sources_freq[:, :, 0] = 0
    sources_freq[:, :, (int(lp * time_dim)):] = 0
    sources_filt = irfft(sources_freq, n=time_dim)
    return (
        (sources_filt.T - sources_filt.T.mean(0)) /
        sources_filt.T.std(0)
    ).squeeze().T


def mix_data_01(ts, mixture_dim=9, return_mix_matrix=False, seed=None):
    np.random.seed(seed)
    signal_dim = ts.shape[-2]
    mix = np.random.randint(-1, 2, (mixture_dim, signal_dim))
    if return_mix_matrix:
        return (mix @ ts.T).T, mix
    return (mix @ ts.T).T


def mix_card_probs_poisson_normalised(latent_dim, mu=1, loc=1):
    base = poisson.pmf(np.arange(latent_dim), mu=mu, loc=loc)
    return base / base.sum()


def choose_mix_card(latent_dim, observed_dim, probs):
    return np.random.choice(latent_dim, size=(observed_dim,), p=probs)


def create_mixture_matrix(
        observed_dim,
        latent_dim,
        mix_probs=None
    ):
    mask = np.zeros((observed_dim, latent_dim))
    mix_probs = mix_probs or mix_card_probs_poisson_normalised(
        latent_dim, mu=(max(1, latent_dim // 3))
    )
    mix_card = choose_mix_card(latent_dim, observed_dim, mix_probs)
    for i, n_signals in enumerate(mix_card):
        idx = np.random.permutation(latent_dim)[:n_signals]
        mask[i, idx] = 1
    weights = np.random.randn(observed_dim, latent_dim)
    state = mask * weights
    return state / np.abs(state).sum(-1, keepdims=True)


def mix_data(mixture, sources, local=None, local_scale=0.25):
    src_signals = mixture @ sources
    local_signals = 0
    if local is not None:
        local_weights = local_scale * np.random.randn(*local.shape[:-1], 1)
        local_signals = local_weights * local
    return src_signals + local_signals


def synthesise_mixture(
        time_dim=200,
        observed_dim=9,
        latent_dim=100,
        subject_dim=1,
        include_local=False,
        local_scale=0.25,
        lp=0.3,
        seed=0,
        mixture=None,
        return_mix_matrix=False
    ):
    np.random.seed(seed)
    sources = synth_slow_signals(
        signal_dim=latent_dim,
        time_dim=time_dim,
        subject_dim=subject_dim,
        lp=lp,
        seed=seed
    )
    if include_local:
        local = synth_slow_signals(
            signal_dim=observed_dim,
            time_dim=time_dim,
            subject_dim=subject_dim,
            lp=lp,
            seed=(seed + 1)
        )
    else:
        local = None
    if mixture is None:
        mixture = create_mixture_matrix(
            observed_dim=observed_dim,
            latent_dim=latent_dim
        )
    signals = mix_data(
        mixture=mixture,
        sources=sources,
        local=local,
        local_scale=local_scale
    )
    if return_mix_matrix:
        return signals, mixture
    return signals
