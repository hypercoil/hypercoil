# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for specialised matrix operations
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from scipy.linalg import toeplitz as toeplitz_ref
from scipy.sparse import csr_matrix
from hypercoil.functional import (
    cholesky_invert,
    toeplitz,
    symmetric,
    spd,
    expand_outer,
    recondition_eigenspaces,
    delete_diagonal,
    fill_diagonal,
    sym2vec,
    vec2sym,
    squareform
)
from hypercoil.functional.utils import vmap_over_outer


class TestMatrix:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.tol = 5e-3
        self.approx = lambda out, ref: np.allclose(out, ref, atol=self.tol)
        np.random.seed(10)


        A = np.random.rand(10, 10)
        self.A = A @ A.T
        self.c = np.random.rand(3)
        self.r = np.random.rand(3)
        self.r[0] = self.c[0]
        self.C = np.random.rand(3, 3)
        self.R = np.random.rand(3, 4)
        self.R[:, 0] = self.C[:, 0]
        self.f = 2
        self.B = np.random.rand(20, 10, 10)
        BLR = np.random.rand(20, 10, 2)
        self.BLR = BLR @ BLR.swapaxes(-1, -2)

    def test_cholesky_invert(self):
        out = cholesky_invert(cholesky_invert(self.A))
        ref = self.A
        assert np.allclose(out, ref, atol=1e-2)

    def test_symmetric(self):
        out = symmetric(self.B)
        ref = out.swapaxes(-1, -2)
        assert self.approx(out, ref)

    def test_expand_outer(self):
        L = np.random.rand(8, 4, 10, 3)
        C = -np.random.rand(8, 4, 3, 1)
        out = expand_outer(L, C=C)
        assert np.all(out <= 0)
        C = vmap_over_outer(jnp.diagflat, 1)((C.squeeze(),))
        out2 = expand_outer(L, C=C)
        assert np.allclose(out, out2)

        out3 = jax.jit(expand_outer)(L, C=C)
        assert np.allclose(out, out3)

        L = np.random.rand(10)
        R = np.random.rand(10)
        out = expand_outer(L, R, symmetry='cross')
        ref = np.outer(L, R)
        ref = (ref + ref.T) / 2
        assert np.allclose(out, ref)

    def test_spd(self):
        out = spd(self.B)
        ref = out.swapaxes(-1, -2)
        assert self.approx(out, ref)
        L = np.linalg.eigvalsh(out)
        assert np.all(L > 0)

        out = spd(self.B, method='svd')
        ref = out.swapaxes(-1, -2)
        assert self.approx(out, ref)
        L = np.linalg.eigvalsh(out)
        assert np.all(L > 0)

    def test_spd_singular(self):
        out = spd(self.BLR, method='eig')
        ref = out.swapaxes(-1, -2)
        assert self.approx(out, ref)
        L = jnp.linalg.eigvalsh(out)
        assert np.all(L > 0)

    def test_toeplitz(self):
        out = toeplitz(self.c, self.r)
        ref = toeplitz_ref(self.c, self.r)
        assert self.approx(out, ref)

    def test_toeplitz_stack(self):
        C = np.random.rand(3, 10)
        R = np.random.rand(2, 3, 8)
        out = toeplitz(C, R)
        assert toeplitz(C, R).shape == (2, 3, 10, 8)

        out = jax.jit(toeplitz)(C, R)
        assert toeplitz(C, R).shape == (2, 3, 10, 8)

        rr = R[0, 1]
        cc = C[1]
        ref = toeplitz(cc, rr)
        assert self.approx(out[0, 1], ref)

        out = toeplitz(self.C, self.R)
        ref = np.stack([toeplitz_ref(c, r) for c, r in zip(self.C, self.R)],
                       axis=0)
        assert self.approx(out, ref)

    def test_toeplitz_extend(self):
        shape = (10, 8)
        out = toeplitz(self.C, self.R, shape=shape)
        assert out.shape == (3, 10, 8)
        Cx, Rx = (np.zeros((self.C.shape[0], shape[0])),
                  np.zeros((self.R.shape[0], shape[1])))
        Cx[:, :self.C.shape[-1]] = self.C
        Rx[:, :self.R.shape[-1]] = self.R
        ref = np.stack([toeplitz_ref(c, r) for c, r in zip(Cx, Rx)])
        assert self.approx(out, ref)

    def test_toeplitz_fill(self):
        shape = (8, 8)
        out = toeplitz(self.C, self.R, shape=shape, fill_value=self.f)
        assert out.shape == (3, 8, 8)
        #out = toeplitz(self.C, self.R, shape=shape, fill_value=self.f)
        Cx, Rx = (np.zeros((self.C.shape[0], shape[0])) + self.f,
                  np.zeros((self.R.shape[0], shape[1])) + self.f)
        Cx[:, :self.C.shape[-1]] = self.C
        Rx[:, :self.R.shape[-1]] = self.R
        ref = np.stack([toeplitz_ref(c, r) for c, r in zip(Cx, Rx)])
        assert self.approx(out, ref)

    def test_recondition(self):
        key = jax.random.PRNGKey(np.random.randint(0, 2**32))
        V = jnp.ones((7, 3))
        d = d = jax.grad(
            lambda X: jnp.linalg.svd(X, full_matrices=False)[0].sum())
        assert np.all(np.isnan(d(V @ V.T)))

        arg = recondition_eigenspaces(V @ V.T, key=key, psi=1e-3, xi=1e-3)
        assert np.logical_not(np.any(np.isnan(d(arg))))

    def test_sym2vec_correct(self):
        from scipy.spatial.distance import squareform
        K = symmetric(np.random.rand(3, 4, 5, 5))
        out = sym2vec(K)

        ref = np.stack([
            np.stack([
                squareform(j * (1 - np.eye(j.shape[0])))
                for j in k
            ]) for k in K
        ])
        assert np.allclose(out, ref)

    def test_fill_diag(self):
        d = 6
        key = jax.random.PRNGKey(np.random.randint(0, 2**32))
        A = jax.random.uniform(key=key, shape=(2, 2, 2, d, d))
        A_fd = fill_diagonal(A)
        A_dd = delete_diagonal(A)
        assert np.allclose(A_fd, A_dd)
        A_fd = fill_diagonal(A, 5)
        assert np.allclose(A_fd, A_dd + 5 * np.eye(d))

        grad = jax.grad(lambda A: fill_diagonal(3 * A, 4).sum())(A)
        grad_ref = 3. * ~np.eye(d, dtype=bool)
        assert np.allclose(grad, grad_ref)

        d2 = 3
        A = jax.random.uniform(key=key, shape=(3, 1, 4, d, d2))
        A_fd = fill_diagonal(A, offset=-1, fill=float('nan'))
        ref = jnp.diagflat(jnp.ones(d, dtype=bool), k=-1)
        ref = ref[:d, :d2]
        assert np.all(np.isnan(A_fd.sum((0, 1, 2))) == ref)

    def test_sym2vec_inversion(self):
        K = symmetric(np.random.rand(3, 4, 5, 5))
        out = vec2sym(sym2vec(K, offset=0), offset=0)
        assert np.allclose(out, K)

    def test_squareform_equivalence(self):
        K = symmetric(np.random.rand(3, 4, 5, 5))
        out = squareform(K)
        ref = sym2vec(K)
        assert np.allclose(out, ref)

        out = squareform(out)
        ref = vec2sym(ref)
        assert np.allclose(out, ref)
