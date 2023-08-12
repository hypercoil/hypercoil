# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Denoising synthesis
~~~~~~~~~~~~~~~~~~~
Synthesise some simple ground truth datasets to test denoising.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Optional, Tuple

from hypercoil.engine.paramutil import PyTree, Tensor, _to_jax_array
from .mix import (
    synth_slow_signals,
    synthesise_mixture
)


def synthesise_artefact(
    time_dim: int = 1000,
    observed_dim: int = 20,
    latent_dim: int = 30,
    subject_dim: int = 100,
    correlated_artefact: bool = False,
    lp: float = 0.3,
    jitter: Tuple[float, float, float] = (0.1, 0.5, 1.5),
    include: Tuple[float, float, float] = (1., 1., 1.),
    spatial_heterogeneity: bool = False,
    subject_heterogeneity: bool = False,
    noise_scale: int = 2,
    *,
    key: 'jax.random.PRNGKey' = None,
):
    key_a, key_j, key_b = jax.random.split(key, 3)
    if correlated_artefact:
        N = synthesise_mixture(
            time_dim=time_dim,
            observed_dim=observed_dim,
            latent_dim=latent_dim,
            subject_dim=subject_dim,
            lp=lp,
            key=key_a
        )
    else:
        N = synth_slow_signals(
            signal_dim=observed_dim,
            time_dim=time_dim,
            subject_dim=subject_dim,
            lp=lp,
            key=key_a
        )

    key_j0, key_j1, key_j2 = jax.random.split(key_j, 3)
    noise_level = jnp.linspace(0, 1, subject_dim)
    jitter = (
        jitter[0] * jax.random.normal(key_j0, (subject_dim,)),
        jitter[1] * jax.random.normal(key_j1, (subject_dim,)),
        jitter[2] * jax.random.normal(key_j2, (subject_dim,)),
    )

    artefact_corrs = (
        jnp.corrcoef(noise_level, noise_level + jitter[0])[1, 0],
        jnp.corrcoef(noise_level, noise_level + jitter[1])[1, 0],
        jnp.corrcoef(noise_level, noise_level + jitter[2])[1, 0],
    )
    print(f'Artefacts synthesised with approx. '
          f'noise level correlations {artefact_corrs}')
    artefacts = (
        include[0] * N[:, 0, :] * (noise_level + jitter[0]).reshape(-1, 1)
        + include[1] * N[:, 1, :] * (noise_level + jitter[1]).reshape(-1, 1)
        + include[2] * N[:, 2, :] * (noise_level + jitter[2]).reshape(-1, 1)
    )
    artefacts = (artefacts - artefacts.mean()) / artefacts.std()

    space = 1
    subj = 1
    if spatial_heterogeneity:
        space = observed_dim
    if subject_heterogeneity:
        subj = subject_dim
    if not subject_heterogeneity and not spatial_heterogeneity:
        betas = 1
    else:
        betas = jax.random.uniform(key_b, (subj, space, 1))
    return (
        N,
        noise_scale * betas * artefacts.reshape(subject_dim, 1, time_dim),
        noise_level
    )


def plot_all(
    X: Tensor,
    n_subj: int = 100,
    cor: bool = True,
    save: Optional[str] = None,
):
    n_sqrt = int(jnp.ceil(jnp.sqrt(n_subj)))
    plt.figure(figsize=(n_sqrt, n_sqrt))
    for i in range(n_subj):
        cur = X[i]
        if not cor:
            cur = jnp.corrcoef(cur)
        plt.subplot(n_sqrt, n_sqrt, i + 1)
        plt.imshow(cur, cmap='coolwarm', vmin=-0.5, vmax=0.5)
        plt.xticks([])
        plt.yticks([])
    if save:
        plt.savefig(save, bbox_inches='tight')


def plot_select(model: PyTree, save: Optional[str] = None):
    plt.figure(figsize=(10, 1))
    plt.imshow(_to_jax_array(model.weight).T, cmap='bone', vmin=0)
    plt.xticks([])
    plt.yticks([])
    if save:
        plt.savefig(save, bbox_inches='tight')


def plot_norm_reduction(w, save=None):
    W = w.T
    n_confounds = w.shape[-1]
    I = jnp.eye(n_confounds)
    theta, _, _, _ = jnp.linalg.lstsq(W, I)
    residual = (I - (W @ theta))

    plt.figure(figsize=(6, 6))
    plt.bar(
        jnp.arange(20),
        (
            jnp.linalg.norm(I, axis=1) -
            jnp.linalg.norm(residual, axis=1)
        ),
        color='black'
    )
    plt.title('Norm Reduction')
    plt.xticks([])
    if save:
        plt.savefig(save, bbox_inches='tight')
