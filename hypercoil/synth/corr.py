# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data synthesis: covariance
~~~~~~~~~~~~~~~~~~~~~~~~~~
Synthesise a dataset for experimental tests of the covariance module.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from hypercoil.functional.cov import corr, pairedcorr
from hypercoil.functional.matrix import toeplitz, sym2vec
from .mix import (
    synthesise_mixture,
    create_mixture_matrix,
    synth_slow_signals
)


DEFAULT_TRANSITION = torch.tensor([
    [0.95, 0.04, 0.01,    0,    0,    0],
    [0.01, 0.95, 0.01, 0.03,    0,    0],
    [0.01,    0, 0.95, 0.03, 0.01,    0],
    [   0,    0,    0, 0.93, 0.03, 0.04],
    [0.03,    0, 0.01,    0, 0.95, 0.01],
    [   0, 0.01, 0.02, 0.01, 0.01, 0.95]
])


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


def synthesise_state_markov_chain(
    time_dim=1000,
    subject_dim=100,
    latent_dim=10,
    observed_dim=30,
    state_weight=1,
    subject_weight=1,
    n_states=6,
    transition_matrix=None,
    begin_states=None,
    seed=1,
):
    np.random.seed(seed)
    state_mix = [create_mixture_matrix(
        observed_dim=observed_dim,
        latent_dim=latent_dim
    ) for _ in range(n_states)]
    subject_mix = [create_mixture_matrix(
        observed_dim=observed_dim,
        latent_dim=latent_dim
    ) for _ in range(subject_dim)]

    all_mixtures = (
        subject_weight * np.array(subject_mix).reshape(
            1, subject_dim, observed_dim, latent_dim) +
        state_weight * np.array(state_mix).reshape(
            n_states, 1, observed_dim, latent_dim)
    )

    if transition_matrix is None: transition_matrix = DEFAULT_TRANSITION

    srcmat = simulate_markov_transitions(
        transition_matrix=transition_matrix,
        time_dim=time_dim,
        subject_dim=subject_dim,
        n_states=n_states,
        begin_states=begin_states,
        seed=seed
    )

    active_state = np.eye(n_states)[srcmat.astype(int)]
    active_state = gaussian_filter1d(active_state, sigma=2, axis=-2)

    latents = synth_slow_signals(
        signal_dim=latent_dim,
        time_dim=time_dim,
        subject_dim=subject_dim,
        seed=(seed + 1)
    )
    all_state_ts = all_mixtures @ latents
    observed_ts = (
        all_state_ts *
        np.expand_dims(active_state.transpose(2, 0, 1), -2)
    ).sum(0)

    return observed_ts, srcmat, state_mix, active_state


def simulate_markov_transitions(
    transition_matrix,
    time_dim=1000,
    subject_dim=100,
    n_states=6,
    begin_states=None,
    seed=None
):
    if seed is not None: torch.manual_seed(seed)
    np.random.seed(seed)
    if begin_states is None:
        begin_states = np.random.randint(0, n_states, subject_dim)
    srcmat = np.zeros((subject_dim, time_dim))
    srcmat[:, 0] = begin_states
    for s in range(subject_dim):
        current_state = int(srcmat[s, 0])
        for t in range(1, time_dim):
            probs = transition_matrix[current_state]
            distr = torch.distributions.Categorical(probs)
            srcmat[s, t] = distr.sample()
            current_state = int(srcmat[s, t])
    return srcmat


def state_presence(time_dim, n_states):
    q = time_dim // n_states + 1
    srcmat = np.tile(np.arange(n_states), (q, 1))
    active_state = srcmat.T.ravel()[:time_dim]
    active_state = np.eye(n_states)[active_state]
    return gaussian_filter1d(active_state, sigma=10, axis=-2)


def overall_signal(signals, weights):
    signal = (signals * weights).sum(-1)
    return (
        (signal - signal.mean(-1, keepdims=True)) /
        signal.std(-1, keepdims=True)
    )


def plot_states(cors, save=None, lim=0.4, vpct=None):
    n_states = cors.shape[0]
    plt.figure(figsize=(3 * n_states, 3))
    for i, s in enumerate(cors):
        plt.subplot(1, 6, i + 1)
        if vpct is not None:
            lim = np.quantile(s - np.eye(s.shape[-1]), vpct)
        plt.imshow(s, cmap='coolwarm', vmin=-lim, vmax=lim)
        plt.xticks([])
        plt.yticks([])
    if save is not None:
        plt.savefig(save, bbox_inches='tight')


def plot_state_ts(state_ts, save=None):
    h, w = state_ts.shape
    aspect = 100 * h / w / 4
    plt.figure(figsize=(20, 5))
    plt.imshow(state_ts, cmap='magma', aspect=aspect)
    plt.xticks([])
    plt.yticks([])
    if save is not None:
        plt.savefig(save, bbox_inches='tight')


def get_transitions(state_ts):
    return np.where(np.diff(state_ts) != 0)[1]


def plot_state_transitions(state_ts, transitions=None, save=None):
    if transitions is None: transitions = []
    plt.figure(figsize=(16, 4))
    plt.plot(state_ts)
    plt.ylim([0, 1])
    for t in transitions:
        plt.axvline(t, ls=':', color='grey')
    if save is not None:
        plt.savefig(save, bbox_inches='tight')


def correlation_alignment(X, X_hat, n_states):
    aligncorr = pairedcorr(X, X_hat).detach().numpy()
    alignments = [None for _ in range(n_states)]
    while aligncorr.sum() > 0:
        x, x_hat = np.unravel_index(aligncorr.argmax(), aligncorr.shape)
        alignments[x_hat] = (x, aligncorr[x, x_hat])
        aligncorr[x, :] = 0
        aligncorr[:, x_hat] = 0
    realigned = X[list(zip(*alignments))[0], :]
    return realigned


def sliding_window_weight(window_length, step_size, time_dim):
    full = toeplitz(
        c=torch.tensor([1 for _ in range(window_length)]),
        r=torch.tensor([1]),
        dim=(time_dim, time_dim)
    )
    sliding = full[(window_length - 1):, :]
    step = slice(0, None, step_size)
    sliding = sliding[step, :]
    return sliding


def kmeans_init(
    X,
    n_states=6,
    window_length=50,
    step_size=10,
    subject_dim=100,
    time_dim=1000
):
    from scipy.cluster.vq import kmeans
    sliding = sliding_window_weight(
        window_length=window_length,
        step_size=step_size,
        time_dim=time_dim
    )
    swc = corr(X, weight=sliding.unsqueeze(-2))
    centroids, error = kmeans(
        sym2vec(swc).view(subject_dim * sliding.shape[0], -1),
        k_or_guess=n_states
    )
    return torch.tensor(centroids).type(X.dtype).to(X.device)