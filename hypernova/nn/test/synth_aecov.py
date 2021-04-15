# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data synthesis: covariance
~~~~~~~~~~~~~~~~~~~~~~~~~~
Synthesise a dataset for the covariance autoencoder test.
"""
import numpy as np
from scipy.fft import rfft, irfft
from scipy.stats import poisson
from scipy.special import softmax
from scipy.ndimage import gaussian_filter1d


def synthesise(n=1000, d=7, p=100, k=2, seed=0):
	np.random.seed(seed)
	sources = slow_signals(d, n)
	local = slow_signals(p, n)
	states = [create_state(d, p) for _ in range(k)]
	state_time = state_presence(n, k)
	sigs = np.stack(
		[state_signals(s, sources, local) for s in states],
		axis=-1)
	signal = overall_signal(sigs, state_time)
	statevar = [np.corrcoef(s[:, :-1]) for s in states]
	return signal, statevar, state_time


def slow_signals(d, n):
    sources = np.random.rand(d, n)
    sources_fft = rfft(sources, n=n)
    sources_fft[:, 0] = 0
    sources_fft[:, 20:] = 0
    sources_filt = irfft(sources_fft, n=n)
    return (
    	(sources_filt.T - sources_filt.T.mean(0)) / sources_filt.T.std(0)).T


def mix_card_probs_pn(d):
    base = poisson.pmf(np.arange(d), mu=1, loc=1)
    return base / base.sum()


def choose_mix_card(d, p, probs):
    return np.random.choice(d, size=(p,), p=probs)


def create_state(d, p, mix_probs=None):
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


def state_presence(n, k):
    q = n // k + 1
    srcmat = np.tile(np.arange(k), (q, 1))
    active_state = srcmat.T.ravel()[:n]
    active_state = np.eye(k)[active_state]
    return gaussian_filter1d(active_state, sigma=10, axis=0)


def state_signals(state, sources, local):
    src_signals = state[:, :-1] @ sources
    loc_signals = state[:, -1] * local.T
    return src_signals + loc_signals.T


def overall_signal(signals, weights):
    signal = (signals * weights).sum(-1)
    return (signal - signal.mean(-1, keepdims=True)) / signal.std(-1, keepdims=True)


