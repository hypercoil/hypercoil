# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data synthesis: covariance
~~~~~~~~~~~~~~~~~~~~~~~~~~
Synthesise a dataset for the covariance autoencoder test.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .mix import synthesise_mixture


def incr_seed(base, incr):
    if base is None:
        return None
    return base + incr


def synthesise_state_transition(
    time_dim=1000,
    latent_dim=7,
    observed_dim=100,
    n_states=2,
    state_timing='equipartition',
    seed=None
):
    mixes = [
        synthesise_mixture(
            time_dim=time_dim,
            observed_dim=observed_dim,
            latent_dim=latent_dim,
            subject_dim=1,
            include_local=True,
            local_scale=0.25,
            lp=0.3,
            hp=0.0,
            seed=incr_seed(seed, s),
            return_mix_matrix=True
        )
        for s in range(n_states)
    ]
    signals, states = list(zip(*mixes))
    signals = np.stack([s.squeeze() for s in signals], axis=-1)
    if state_timing == 'equipartition':
        state_time = state_presence(time_dim=time_dim, n_states=n_states)
    signal = overall_signal(signals, state_time)
    statevar = [np.corrcoef(s[:, :-1]) for s in states]
    return signal, statevar, state_time


def state_presence(time_dim, n_states):
    q = time_dim // n_states + 1
    srcmat = np.tile(np.arange(n_states), (q, 1))
    active_state = srcmat.T.ravel()[:time_dim]
    active_state = np.eye(n_states)[active_state]
    return gaussian_filter1d(active_state, sigma=10, axis=-2)


def overall_signal(signals, weights):
    signal = (signals * weights).sum(-1)
    return (signal - signal.mean(-1, keepdims=True)) / signal.std(-1, keepdims=True)
