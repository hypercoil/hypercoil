# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Denoising synthesis
~~~~~~~~~~~~~~~~~~~
Synthesise some simple ground truth datasets to test denoising.
"""
import numpy as np
import matplotlib.pyplot as plt

from .mix import (
    synth_slow_signals,
    synthesise_mixture
)


def synthesise_artefact(
    time_dim=1000,
    observed_dim=20,
    latent_dim=30,
    subject_dim=100,
    correlated_artefact=False,
    seed=None,
    lp=0.3,
    strong_jitter=0.1,
    moderate_jitter=0.5,
    weak_jitter=1.5,
    spatial_heterogeneity=False,
    subject_heterogeneity=False,
    noise_scale=2
):
    np.random.seed(seed)
    if correlated_artefact:
        N = synthesise_mixture(
            time_dim=time_dim,
            observed_dim=observed_dim,
            latent_dim=latent_dim,
            subject_dim=subject_dim,
            lp=lp,
            seed=seed
        )
    else:
        N = synth_slow_signals(
            signal_dim=observed_dim,
            time_dim=time_dim,
            subject_dim=subject_dim,
            lp=lp,
            seed=seed
        )

    noise_level = np.linspace(0, 1, subject_dim)
    strong = strong_jitter * np.random.randn(subject_dim)
    mod = moderate_jitter * np.random.randn(subject_dim)
    weak = weak_jitter * np.random.randn(subject_dim)

    artefact_corrs = (
        np.corrcoef(noise_level, noise_level + strong)[1, 0],
        np.corrcoef(noise_level, noise_level + mod)[1, 0],
        np.corrcoef(noise_level, noise_level + weak)[1, 0],
    )
    artefacts = (
        N[:, 0, :] * (noise_level + strong).reshape(-1, 1)
        + N[:, 1, :] * (noise_level + mod).reshape(-1, 1)
        + N[:, 2, :] * (noise_level + weak).reshape(-1, 1)
    )

    space = 1
    subj = 1
    if spatial_heterogeneity:
        space = observed_dim
    if subject_heterogeneity:
        subj = subject_dim
    betas = np.random.rand(subj, space, 1)
    return noise_scale * betas * artefacts.reshape(subject_dim, 1, time_dim)


def plot_all(X, n_subj=100, cor=True, save=None):
    n_sqrt = int(np.ceil(np.sqrt(n_subj)))
    plt.figure(figsize=(n_sqrt, n_sqrt))
    for i in range(n_subj):
        cur = X[i]
        if not cor:
            cur = np.corrcoef(cur)
        plt.subplot(n_sqrt, n_sqrt, i + 1)
        plt.imshow(cur, cmap='coolwarm', vmin=-0.5, vmax=0.5)
        plt.xticks([])
        plt.yticks([])
    if save:
        plt.savefig(save)
