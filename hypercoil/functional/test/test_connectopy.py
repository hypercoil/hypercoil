# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for connectopic gradient mapping.
"""
import pytest
import torch
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from brainspace import datasets
from brainspace.gradient.embedding import(
    laplacian_eigenmaps as le_ref,
    diffusion_mapping as dm_ref
)
from pkg_resources import resource_filename as pkgrf
from hypercoil.functional.connectopy import (
    laplacian_eigenmaps,
    diffusion_mapping
)


class TestConnectopy:

    def plot_gradients(self, ref, test, name):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].scatter(ref[0], ref[1])
        ax[0].set_title('Reference')
        ax[1].scatter(test[0], test[1])
        ax[1].set_title('Test')
        results = pkgrf(
            'hypercoil',
            'results/'
        )
        fig.savefig(f'{results}/{name}.png')

    def test_circle_eigenmaps(self):
        dim = 101
        A = np.diagflat(np.ones(dim - 1), 1)
        A[0, -1] = 1
        A = A + A.T
        Q, L = laplacian_eigenmaps(A)
        Q_ref, L_ref = le_ref(A)
        self.plot_gradients(
            ref=Q_ref[..., :2].T,
            test=Q[..., :2].T,
            name='eigenmaps_circle'
        )

    def test_schaefer_eigenmaps(self):
        A = datasets.load_group_fc(parcellation='schaefer')
        Q_ref, L_ref = le_ref(A, norm_laplacian=True)
        Q, L = laplacian_eigenmaps(A)

        self.plot_gradients(
            ref=Q_ref[..., :2].T,
            test=Q[..., :2].T,
            name='eigenmaps_schaefer'
        )
        A_grad = jax.grad(lambda x: laplacian_eigenmaps(x)[0].sum())(A)
        assert A_grad is not None

    def test_circle_diffusion(self):
        dim = 101
        A = np.diagflat(np.ones(dim - 1), 1)
        A[0, -1] = 1
        A = A + A.T
        Q, L = diffusion_mapping(A)
        Q_ref, L_ref = dm_ref(A)
        self.plot_gradients(
            ref=Q_ref[..., :2].T,
            test=Q[..., :2].T,
            name='diffusion_circle'
        )

    def test_schaefer_diffusion(self):
        A = datasets.load_group_fc(parcellation='schaefer')
        Q, L = diffusion_mapping(A)
        Q_ref, L_ref = dm_ref(A)
        self.plot_gradients(
            ref=Q_ref[..., :2].T,
            test=Q[..., :2].T,
            name='diffusion_schaefer'
        )

        Q_ref, L_ref = dm_ref(A, alpha=1, diffusion_time=10)
        Q, L = diffusion_mapping(
            A,
            alpha=1,
            diffusion_time=10
        )
        self.plot_gradients(
            ref=Q_ref[..., :2].T,
            test=Q[..., :2].T,
            name='diffusion_schaefer_param'
        )
        A_grad = jax.grad(lambda x: laplacian_eigenmaps(x)[0].sum())(A)
        assert A_grad is not None
