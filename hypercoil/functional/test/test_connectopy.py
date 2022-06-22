# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for connectopic gradient mapping.
"""
import pytest
import torch
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
        A = torch.diag_embed(torch.ones(dim - 1), 1)
        A[0, -1] = 1
        A = A + A.t()
        Q, L = laplacian_eigenmaps(A)
        Q_ref, L_ref = le_ref(A.detach().numpy())
        self.plot_gradients(
            ref=Q_ref[..., :2].T,
            test=Q[..., :2].t(),
            name='eigenmaps_circle'
        )

    def test_schaefer_eigenmaps(self):
        A = datasets.load_group_fc(parcellation='schaefer')
        Q_ref, L_ref = le_ref(A, norm_laplacian=True)
        A = torch.tensor(A)
        A.requires_grad = True
        Q, L = laplacian_eigenmaps(A)

        assert A.grad is None
        Q.sum().backward()
        assert A.grad is not None
        self.plot_gradients(
            ref=Q_ref[..., :2].T,
            test=Q[..., :2].t().detach(),
            name='eigenmaps_schaefer'
        )

    def test_circle_eigenmaps_sparse(self):
        dim = 101
        W = torch.ones(dim, dtype=torch.float)
        E = list(zip(range(dim - 1), range(1, dim)))
        E += [(100, 0)]
        E = torch.tensor(E)
        Q, L = laplacian_eigenmaps(W, E.t())
        A = csr_matrix((W, E.t()), (dim, dim))
        Q_ref, L_ref = le_ref(A, norm_laplacian=True)
        self.plot_gradients(
            ref=Q_ref[..., :2].T,
            test=Q[..., :2].t(),
            name='eigenmaps_sparse'
        )

    def test_circle_diffusion(self):
        dim = 101
        A = torch.diag_embed(torch.ones(dim - 1), 1)
        A[0, -1] = 1
        A = A + A.t()
        Q, L = diffusion_mapping(A)
        Q_ref, L_ref = dm_ref(A.numpy())
        self.plot_gradients(
            ref=Q_ref[..., :2].T,
            test=Q[..., :2].t(),
            name='diffusion_circle'
        )

    def test_schaefer_diffusion(self):
        A = datasets.load_group_fc(parcellation='schaefer')
        Q, L = diffusion_mapping(torch.tensor(A), solver='eigh')
        Q_ref, L_ref = dm_ref(A)
        self.plot_gradients(
            ref=Q_ref[..., :2].T,
            test=Q[..., :2].t(),
            name='diffusion_schaefer'
        )

        Q_ref, L_ref = dm_ref(A, alpha=1, diffusion_time=10)
        A = torch.tensor(A)
        A.requires_grad = True
        Q, L = diffusion_mapping(
            A,
            alpha=1,
            diffusion_time=10
        )

        assert A.grad is None
        Q.sum().backward()
        assert A.grad is not None
        self.plot_gradients(
            ref=Q_ref[..., :2].T,
            test=Q[..., :2].t().detach(),
            name='diffusion_schaefer_param'
        )

    def test_circle_diffusion_sparse(self):
        dim = 101
        W = torch.ones(dim, dtype=torch.float)
        E = list(zip(range(dim - 1), range(1, dim)))
        E += [(100, 0)]
        E = torch.tensor(E)
        Q, L = diffusion_mapping(W, E.t())
        A = csr_matrix((W, E.t()), (dim, dim))
        Q_ref, L_ref = dm_ref(A)
        self.plot_gradients(
            ref=Q_ref[..., :2].T,
            test=Q[..., :2].t(),
            name='diffusion_sparse'
        )

        W.requires_grad = True
        Q, L = diffusion_mapping(W, E.t(), solver='svd')
        # This eigendecomposition is so degenerate that we should be getting
        # exploding or NaN-valued gradients.
        assert W.grad is None
        Q.sum().backward()
        assert W.grad is not None
        assert (torch.isnan(W.grad.max()) or W.grad.max() > 1e5)
        print(W.grad)
        self.plot_gradients(
            ref=Q_ref[..., :2].T,
            test=Q[..., :2].t().detach(),
            name='diffusion_sparse_svd'
        )
