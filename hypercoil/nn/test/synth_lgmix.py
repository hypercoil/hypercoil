# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Local-global mixture synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Synthesise a dataset by combining 'local' sources with a linear  mixture of
'global' sources.
"""
import numpy as np
from scipy.fft import rfft, irfft
from scipy.stats import poisson
#from scipy.special import softmax


def synthesise(n=1000, d=7, p=100, n_sub=1, seed=0):
    np.random.seed(seed)
    sources = slow_signals(d, n, n_sub)
    local = slow_signals(p, n, n_sub)
    mixture = create_mixture(d, p)
    sigs = observed_signals(mixture, sources, local)
    return sigs, mixture


def slow_signals(d, n, n_sub=1, lp_bin=20):
    sources = np.random.rand(n_sub, d, n)
    sources_fft = rfft(sources, n=n)
    sources_fft[:, :, 0] = 0
    sources_fft[:, :, lp_bin:] = 0
    sources_filt = irfft(sources_fft, n=n)
    return (
        (sources_filt.T - sources_filt.T.mean(0)) / sources_filt.T.std(0)).T


def mix_card_probs_pn(d):
    base = poisson.pmf(np.arange(d), mu=1, loc=1)
    return base / base.sum()


def choose_mix_card(d, p, probs):
    return np.random.choice(d, size=(p,), p=probs)


def create_mixture(d, p, mix_probs=None):
    mask = np.zeros((p, d + 1))
    mask[:, -1] = 1
    mix_probs = mix_probs or mix_card_probs_pn(d)
    mix_card = choose_mix_card(d, p, mix_probs)
    for i, n_signals in enumerate(mix_card):
        idx = np.random.permutation(d)[:n_signals]
        mask[i, idx] = 1
    weights = np.random.randn(p, d + 1)
    weights[:, -1] *= 0.25
    state = mask * weights
    #state = softmax(state ** 2, axis=-1) * mask
    return state / np.abs(state).sum(-1, keepdims=True)


def observed_signals(mixture, sources, local):
    src_signals = mixture[:, :-1] @ sources
    loc_signals = mixture[:, -1].reshape(-1, 1) * local
    return src_signals + loc_signals
