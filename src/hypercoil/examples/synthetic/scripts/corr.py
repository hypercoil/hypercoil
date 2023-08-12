# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data synthesis: covariance
~~~~~~~~~~~~~~~~~~~~~~~~~~
Synthesise a dataset for experimental tests of the covariance module.
"""
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Sequence
from scipy.ndimage import gaussian_filter1d
from hypercoil.engine.paramutil import Tensor
from hypercoil.functional.cov import corr, pairedcorr
from hypercoil.functional.matrix import toeplitz, sym2vec
from .mix import (
    synthesise_mixture,
    create_mixture_matrix,
    synth_slow_signals
)


def get_default_transition_matrix() -> Tensor:
    return jnp.log(jnp.array([
        [0.95, 0.04, 0.01,    0,    0,    0],
        [0.01, 0.95, 0.01, 0.03,    0,    0],
        [0.01,    0, 0.95, 0.03, 0.01,    0],
        [   0,    0,    0, 0.93, 0.03, 0.04],
        [0.03,    0, 0.01,    0, 0.95, 0.01],
        [   0, 0.01, 0.02, 0.01, 0.01, 0.95]
    ]))


def synthesise_state_transition(
    time_dim: int = 1000,
    latent_dim: int = 7,
    observed_dim: int = 100,
    n_states: int = 2,
    state_timing: Optional[str] = 'equipartition',
    *,
    key: 'jax.random.PRNGKey',
):
    keys = jax.random.split(key, n_states)
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
            return_mix_matrix=True,
            key=k,
        )
        for k in keys
    ]
    signals, states = list(zip(*mixes))
    signals = jnp.stack([s.squeeze() for s in signals], axis=-1)
    if state_timing == 'equipartition':
        state_time = state_presence(time_dim=time_dim, n_states=n_states)
    signal = overall_signal(signals, state_time)
    statevar = [jnp.corrcoef(s[:, :-1]) for s in states]
    return signal, statevar, state_time


def synthesise_state_markov_chain(
    time_dim: int = 1000,
    subject_dim: int = 100,
    latent_dim: int = 10,
    observed_dim: int = 30,
    state_weight: float = 1.,
    subject_weight: float = 1.,
    n_states: int = 6,
    transition_matrix: Tensor = None,
    begin_states: Optional[Tensor] = None,
    *,
    key: 'jax.random.PRNGKey',
):
    key_st, key_su, key_t, key_l = jax.random.split(key, 4)
    keys_st = jax.random.split(key_st, n_states)
    keys_su = jax.random.split(key_su, subject_dim)
    state_mix = [create_mixture_matrix(
        observed_dim=observed_dim,
        latent_dim=latent_dim,
        key=k,
    ) for k in keys_st]
    subject_mix = [create_mixture_matrix(
        observed_dim=observed_dim,
        latent_dim=latent_dim,
        key=k,
    ) for k in keys_su]

    all_mixtures = (
        subject_weight * jnp.array(subject_mix).reshape(
            1, subject_dim, observed_dim, latent_dim) +
        state_weight * jnp.array(state_mix).reshape(
            n_states, 1, observed_dim, latent_dim)
    )

    if transition_matrix is None:
        transition_matrix = get_default_transition_matrix()

    srcmat = simulate_markov_transitions(
        transition_matrix=transition_matrix,
        time_dim=time_dim,
        subject_dim=subject_dim,
        n_states=n_states,
        begin_states=begin_states,
        key=key_t,
    )

    active_state = jnp.eye(n_states)[srcmat.astype(int)]
    active_state = jnp.array(
        gaussian_filter1d(active_state, sigma=2, axis=-2))

    latents = synth_slow_signals(
        signal_dim=latent_dim,
        time_dim=time_dim,
        subject_dim=subject_dim,
        key=key_l
    )
    all_state_ts = all_mixtures @ latents
    observed_ts = (
        all_state_ts *
        jnp.expand_dims(active_state.transpose(2, 0, 1), -2)
    ).sum(0)

    return observed_ts, srcmat, state_mix, active_state


def simulate_markov_transitions(
    transition_matrix: Tensor,
    time_dim: int = 1000,
    subject_dim: int = 100,
    n_states: int = 6,
    begin_states: Optional[Tensor] = None,
    *,
    key: 'jax.random.PRNGKey',
) -> Tensor:
    key_s, key_p = jax.random.split(key, 2)
    if begin_states is None:
        begin_states = jax.random.randint(key_s, (subject_dim,), 0, n_states)
    srcmat = jnp.zeros((subject_dim, time_dim), dtype=int)
    srcmat = srcmat.at[:, 0].set(begin_states)
    for t in range(1, time_dim):
        #print(transition_matrix[srcmat[:, t-1]])
        key = jax.random.fold_in(key_p, t)
        srcmat = srcmat.at[:, t].set(
            jax.random.categorical(key, transition_matrix[srcmat[:, t-1]]))
    return srcmat


def state_presence(
    time_dim: int,
    n_states: int,
) -> Tensor:
    q = time_dim // n_states + 1
    srcmat = jnp.tile(jnp.arange(n_states), (q, 1))
    active_state = srcmat.T.ravel()[:time_dim]
    active_state = jnp.eye(n_states)[active_state]
    return jnp.array(gaussian_filter1d(active_state, sigma=10, axis=-2))


def overall_signal(
    signals: Tensor,
    weights: Tensor,
) -> Tensor:
    signal = (signals * weights).sum(-1)
    return (
        (signal - signal.mean(-1, keepdims=True)) /
        signal.std(-1, keepdims=True)
    )


def plot_states(
    cors: Tensor,
    save: Optional[str] = None,
    lim: float = 0.4,
    vpct: Optional[float] = None
):
    n_states = cors.shape[0]
    plt.figure(figsize=(3 * n_states, 3))
    for i, s in enumerate(cors):
        plt.subplot(1, 6, i + 1)
        if vpct is not None:
            lim = jnp.quantile(s - jnp.eye(s.shape[-1]), vpct)
        plt.imshow(s, cmap='coolwarm', vmin=-lim, vmax=lim)
        plt.xticks([])
        plt.yticks([])
    if save is not None:
        plt.savefig(save, bbox_inches='tight')


def plot_state_ts(
    state_ts: Tensor,
    save: Optional[str] = None
):
    plt.figure(figsize=(20, 5))
    plt.imshow(state_ts, cmap='magma', aspect='auto', interpolation='nearest')
    plt.xticks([])
    plt.yticks([])
    if save is not None:
        plt.savefig(save, bbox_inches='tight')


def get_transitions(state_ts: Tensor) -> Tensor:
    return jnp.where(jnp.diff(state_ts) != 0)[1]


def plot_state_transitions(
    state_ts: Tensor,
    transitions: Optional[Sequence] = None,
    save: Optional[str] = None,
):
    if transitions is None: transitions = []
    plt.figure(figsize=(16, 4))
    plt.plot(state_ts.squeeze())
    plt.ylim([0, 1])
    for t in transitions:
        plt.axvline(t, ls=':', color='grey')
    if save is not None:
        plt.savefig(save, bbox_inches='tight')


def correlation_alignment(X: Tensor, X_hat: Tensor, n_states: int) -> Tensor:
    aligncorr = pairedcorr(X, X_hat)
    alignments = [None for _ in range(n_states)]
    while aligncorr.sum() > 0:
        x, x_hat = jnp.unravel_index(
            aligncorr.argmax(), aligncorr.shape)
        alignments[x_hat] = (x, aligncorr[x, x_hat])
        aligncorr = aligncorr.at[x, :].set(0)
        aligncorr = aligncorr.at[:, x_hat].set(0)
    realigned = X[list(zip(*alignments))[0], :]
    return realigned


def sliding_window_weight(
    window_length: int,
    step_size: int,
    time_dim: int,
) -> Tensor:
    full = toeplitz(
        c=jnp.array([1 for _ in range(window_length)]),
        r=jnp.array([1]),
        shape=(time_dim, time_dim)
    )
    sliding = full[(window_length - 1):, :]
    step = slice(0, None, step_size)
    sliding = sliding[step, :]
    return sliding


def kmeans_init(
    X: Tensor,
    n_states: int = 6,
    window_length: int = 50,
    step_size: int = 10,
    subject_dim: int = 100,
    time_dim: int = 1000,
    *,
    key: 'jax.random.PRNGKey',
):
    from scipy.cluster.vq import kmeans
    seed = jax.random.randint(key, (subject_dim,), 0, 2**31 - 1)
    np.random.seed(seed)
    sliding = sliding_window_weight(
        window_length=window_length,
        step_size=step_size,
        time_dim=time_dim
    )
    swc = corr(X, weight=sliding[..., None, :])
    centroids, error = kmeans(
        sym2vec(swc).reshape((subject_dim * sliding.shape[0], -1)),
        k_or_guess=n_states
    )
    return jnp.array(centroids)
