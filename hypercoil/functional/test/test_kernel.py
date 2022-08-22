# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for kernels and distances.
"""
import torch
import jax
import numpy as np
from hypercoil.functional import (
    linear_kernel,
    polynomial_kernel,
    gaussian_kernel,
    rbf_kernel,
    sigmoid_kernel,
    cosine_kernel
)
from hypercoil.functional.kernel import _param_norm
from hypercoil.functional.sparse import random_sparse
from sklearn.metrics.pairwise import (
    linear_kernel as lk_ref,
    polynomial_kernel as pk_ref,
    rbf_kernel as gk_ref,
    sigmoid_kernel as sk_ref,
    cosine_similarity as ck_ref
)


class TestKernel:

    def random_sparse_input(dim, nse):
        from jax.experimental.sparse import BCOO
        W = np.random.randn(*dim[:-2], nse)
        r = np.random.randint(dim[-2], (nse,))
        c = np.random.randint(dim[-1], (nse,))
        E = np.stack((r, c))[None, ...]
        return BCOO((W, E), shape=dim, n_batch=1)
        #return torch.sparse_coo_tensor(E, W, size=dim)

    def test_linear_kernel(self):
        n, p = 30, 100
        X = np.random.randn(10, n, p)
        Y = np.random.randn(10, n, p)
        ref = np.stack([lk_ref(x) for x in X])
        out = linear_kernel(X)
        assert np.allclose(out, ref, atol=1e-5)
        ref = np.stack([lk_ref(x, y) for x, y in zip(X, Y)])
        out = linear_kernel(X, Y)
        assert np.allclose(out, ref, atol=1e-5)

        linear_kernel_jit = jax.jit(linear_kernel)
        out = linear_kernel_jit(X, Y)
        assert np.allclose(out, ref, atol=1e-5)
        out = linear_kernel_jit(X)
        ref = linear_kernel_jit(X, X)
        assert np.allclose(out, ref, atol=1e-5)

    def test_polynomial_kernel(self):
        n, p = 30, 100
        X = np.random.randn(n, p)
        Y = np.random.randn(n, p)
        ref = linear_kernel(X, Y)
        out = polynomial_kernel(X, Y, gamma=1, order=1)
        assert np.allclose(out, ref, atol=1e-5)
        ref = pk_ref(X, Y)
        out = jax.jit(
            polynomial_kernel,
            static_argnames=('r', 'gamma')
        )(X, Y, r=1)
        assert np.allclose(out, ref, atol=1e-5)
        ref = pk_ref(X, Y, gamma=-1, degree=7, coef0=-100)
        out = polynomial_kernel(X, Y, gamma=-1, order=7, r=-100)
        assert np.allclose(out, ref, atol=1e-5)

    def test_sigmoid_kernel(self):
        n, p = 30, 100
        X = np.random.randn(n, p)
        Y = np.random.randn(n, p)
        ref = sk_ref(X, Y)
        out = jax.jit(
            sigmoid_kernel,
            static_argnames=('r', 'gamma')
        )(X, Y, r=1)
        assert np.allclose(out, ref, atol=1e-5)
        ref = sk_ref(X, Y, gamma=0.71, coef0=-2)
        out = sigmoid_kernel(X, Y, gamma=0.71, r=-2)
        assert np.allclose(out, ref, atol=1e-5)

    def test_gaussian_kernel(self):
        n, p = 30, 100
        X = np.random.randn(n, p)
        Y = np.random.randn(n, p)
        ref = gk_ref(X, Y)
        out = gaussian_kernel(X, Y)
        assert np.allclose(out, ref, atol=1e-5)
        ref = gk_ref(X, Y, gamma=-2e-5)
        out = jax.jit(
            rbf_kernel,
            static_argnames=('gamma',)
        )(X, Y, gamma=-2e-5)
        assert np.allclose(out, ref, atol=1e-5)
        ref = gk_ref(X, Y, gamma=0.25)
        out = jax.jit(
            gaussian_kernel,
            static_argnames=('sigma',)
        )(X, Y, sigma=2)
        assert np.allclose(out, ref, atol=1e-5)

    def test_norm(self):
        X = self.random_sparse_input((20, 20, 3), 30)
        out = _param_norm(X, theta=None).to_dense()
        ref = X.to_dense().norm(dim=1)
        assert torch.allclose(out, ref)

    def test_cosine_kernel(self):
        n, p = 30, 100
        X = torch.randn(n, p)
        Y = torch.randn(n, p)
        ref = ck_ref(X, Y)
        out = cosine_kernel(X, Y)
        assert np.allclose(out, ref, atol=1e-5)

        X = self.random_sparse_input((20, 10, 3), 30)
        Y = self.random_sparse_input((30, 10, 3), 30)
        out = cosine_kernel(X, Y)
        ref = np.stack([ck_ref(x, y) for x, y in zip(
            X.to_dense().permute(-1, 0, 1),
            Y.to_dense().permute(-1, 0, 1))
        ], -1)
        assert out.shape == ref.shape
        assert out._indices().size(-1) * 3 == (ref != 0).sum()
        assert np.allclose(ref, out.to_dense())

    def test_parameterised_kernel(self):
        X = np.array([
            [0., 1., 0., 1., 2.],
            [2., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0.]
        ]).T

        theta = np.array([1., 1., 1.])
        ref = lk_ref(X)
        out = linear_kernel(X, theta=theta)
        assert np.allclose(out, ref, atol=1e-5)

        theta = np.array([1., 1., 0.])
        ref = lk_ref(X)
        out = linear_kernel(X, theta=theta)
        assert np.allclose(out, ref, atol=1e-5)

        theta = np.array([1., 0., 0.])
        ref = (X[:, 0].reshape(-1, 1) @ X[:, 0].reshape(1, -1))
        out = linear_kernel(X, theta=theta)
        assert np.allclose(out, ref, atol=1e-5)

        theta = np.array([0., 1., 0.])
        ref = (X[:, 1].reshape(-1, 1) @ X[:, 1].reshape(1, -1))
        out = linear_kernel(X, theta=theta)
        assert np.allclose(out, ref, atol=1e-5)

        theta = np.array([0., 1., 0.])
        ref = (X[:, 1].reshape(-1, 1) @ X[:, 1].reshape(1, -1))
        out = linear_kernel(X, theta=theta)
        assert np.allclose(out, ref, atol=1e-5)

        theta = np.array([
            [1., 0., 0.],
            [1., 1., 0.]
        ])
        ref = np.stack((
            (X[:, 0].reshape(-1, 1) @ X[:, 0].reshape(1, -1)),
            lk_ref(X),
        ))
        out = linear_kernel(X, theta=theta)
        assert np.allclose(out, ref, atol=1e-5)

        theta = np.eye(3)
        ref = lk_ref(X)
        out = linear_kernel(X, theta=theta)
        assert np.allclose(out, ref, atol=1e-5)

    def test_linear_kernel_sparse(self):
        key = jax.random.PRNGKey(4839)
        k0, k1, k2 = jax.random.split(key, 3)
        X = random_sparse(
            (4, 3, 50, 100),
            k=5,
            key=k0
        )
        Y = random_sparse(
            (4, 3, 100, 100),
            k=5,
            key=k1
        )

        out = linear_kernel(X, Y)
        ref = linear_kernel(X.todense(), Y.todense())
        assert np.allclose(out, ref, atol=1e-5)

        theta = jax.random.normal(k2, shape=(100,))
        out = linear_kernel(X, Y, theta=theta)
        ref = linear_kernel(X.todense(), Y.todense(), theta=theta)
        assert np.allclose(out, ref, atol=1e-5)

        theta = jax.random.normal(k2, shape=(3, 100))
        out = linear_kernel(X, Y, theta=theta)
        ref = linear_kernel(X.todense(), Y.todense(), theta=theta)
        assert np.allclose(out, ref, atol=1e-5)

        theta = random_sparse(
            (100, 100),
            k=5,
            key=k2
        )
        out = linear_kernel(X, Y, theta=theta)
        theta_ref = theta.todense().T @ theta.todense()
        ref = linear_kernel(X.todense(), Y.todense(), theta=theta_ref)
        assert np.allclose(out, ref, atol=1e-5)
