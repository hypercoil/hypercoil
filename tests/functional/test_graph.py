# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for graph and network measures
"""
import jax
import pytest
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from jax.experimental.sparse import BCOO
from hypercoil.functional import (
    modularity_matrix,
    relaxed_modularity,
    graph_laplacian
)
from communities.utilities import (
    modularity_matrix as modularity_matrix_ref,
    modularity as modularity_ref
)

from hypercoil.functional.sparse import random_sparse


#TODO: Missing unit tests:
# - case with positive and negative weights in the adjacency matrix
# - correctness of nonassociative block modularity


class TestGraph:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.tol = 5e-7
        self.approx = lambda out, ref: np.allclose(out, ref, atol=self.tol)

        self.X = np.random.rand(3, 20, 20)
        self.X += self.X.swapaxes(-1, -2)
        self.aff = np.random.randint(0, 4, 20)
        self.comms = [np.where(self.aff==c)[0] for c in np.unique(self.aff)]
        self.C = np.eye(4)[self.aff]
        self.L = np.random.rand(4, 4)

    def test_modularity_matrix(self):
        out = modularity_matrix(self.X, normalise_modularity=True)
        ref = np.stack([modularity_matrix_ref(x) for x in self.X])
        assert self.approx(out, ref)

    def test_modularity(self):
        out = relaxed_modularity(self.X, self.C,
                                 exclude_diag=True,
                                 directed=False)
        ref = np.stack(
            [modularity_ref(modularity_matrix_ref(x), self.comms)
             for x in self.X])
        assert self.approx(out, ref)

    def test_nonassociative_block(self):
        #TODO: this test only checks that the call does not crash
        out = relaxed_modularity(self.X, self.C,
                                 L=self.L, exclude_diag=True) / 2

    def test_laplacian_dense(self):
        A = np.diagflat(np.ones(9), k=1)
        A[0, :] = 0
        A = A + A.T
        Lref = laplacian(A)
        L = graph_laplacian(A, normalise=False)
        assert np.allclose(L, Lref)
        Lref = laplacian(A, normed=True)
        L = graph_laplacian(A)
        assert np.allclose(L[1:, :], Lref[1:, :])
        assert np.allclose(np.diagonal(L), np.ones(10))

    def test_laplacian_topk(self):
        key = jax.random.PRNGKey(0)
        W = random_sparse(
            (4, 3, 100, 100),
            k=5,
            key=key
        )
        L = graph_laplacian(W, normalise=False).todense()
        Lref = graph_laplacian(W.todense(), normalise=False)
        assert np.allclose(L, Lref, atol=1e-5)
        W = BCOO((np.abs(W.data), W.indices), shape=W.shape)
        L = graph_laplacian(W, normalise=True).todense()
        Lref = graph_laplacian(W.todense(), normalise=True)
        assert np.allclose(L, Lref, atol=1e-5)

    def test_laplacian_sparse(self):
        n_nodes = 5
        W = np.random.rand(n_nodes - 2)
        E = np.array(list(zip(range(1, n_nodes - 1), range(2, n_nodes))))
        Esym = np.array(list(zip(
            range(2, n_nodes),
            range(1, n_nodes - 1))))
        E = np.concatenate((E, Esym))
        W = np.concatenate((W, W))
        Ws = BCOO((W, E), shape=(n_nodes, n_nodes))

        L = graph_laplacian(Ws, normalise=True, topk=False)
        L = L.todense()
        Lref = laplacian(
            csr_matrix((W, E.T), (n_nodes, n_nodes)),
            normed=True).todense()

        assert np.allclose(L[1:, :], Lref[1:, :])
        assert np.allclose(np.diag(L), np.ones(n_nodes))
