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
from .synth_lgmix import synthesise as synth_lgmix


def synthesise(n=1000, d=7, p=100, k=2, seed=0):
    lgmixes = [synth_lgmix(n, d, p, 1, seed=(s + seed)) for s in range(k)]
    sigs, states = list(zip(*lgmixes))
    sigs = np.stack([s.squeeze() for s in sigs], axis=-1)
    state_time = state_presence(n, k)
    signal = overall_signal(sigs, state_time)
    statevar = [np.corrcoef(s[:, :-1]) for s in states]
    return signal, statevar, state_time


def state_presence(n, k):
    q = n // k + 1
    srcmat = np.tile(np.arange(k), (q, 1))
    active_state = srcmat.T.ravel()[:n]
    active_state = np.eye(k)[active_state]
    return gaussian_filter1d(active_state, sigma=10, axis=0)


def overall_signal(signals, weights):
    signal = (signals * weights).sum(-1)
    return (signal - signal.mean(-1, keepdims=True)) / signal.std(-1, keepdims=True)
