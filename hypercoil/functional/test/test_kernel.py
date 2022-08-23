# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for kernels and distances.
"""
import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.distance import cdist
from hypercoil.functional import (
    linear_kernel,
    polynomial_kernel,
    gaussian_kernel,
    rbf_kernel,
    sigmoid_kernel,
    cosine_kernel
)
from hypercoil.functional.kernel import param_norm, linear_distance
from hypercoil.functional.sparse import random_sparse
from sklearn.metrics.pairwise import (
    linear_kernel as lk_ref,
    polynomial_kernel as pk_ref,
    rbf_kernel as gk_ref,
    sigmoid_kernel as sk_ref,
    cosine_similarity as ck_ref
)


class TestKernel:

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
        #TODO: We don't have any correctness tests for param_norm yet.
        X = np.random.randn(4, 3, 50, 100)
        out = param_norm(X, squared=True)
        ref = X / (np.linalg.norm(X, axis=-1, keepdims=True) ** 2)
        assert np.allclose(out, ref, atol=1e-5)

        theta = np.random.randn(3, 100, 100)
        theta = theta @ theta.swapaxes(-1, -2)
        out = param_norm(X, theta)
        assert out.shape == X.shape
        theta = np.random.randn(3, 100)
        out = param_norm(X, theta)
        assert out.shape == X.shape

        # dense-sparse equivalence
        key = jax.random.PRNGKey(0)
        X = random_sparse(
            (4, 3, 50, 100),
            k=5,
            key=key
        )
        out = param_norm(X, squared=True).todense()
        ref = param_norm(X.todense(), squared=True)
        assert np.allclose(out, ref, atol=1e-5)

        theta0 = np.random.randn(3, 100, 100)
        theta0 = theta0 @ theta0.swapaxes(-1, -2)
        theta1 = np.random.rand(3, 100)
        for theta in (theta0, theta1,):
            out = param_norm(X, theta).todense()
            ref = param_norm(X.todense(), theta)
            assert np.allclose(out, ref, atol=1e-5)

    def test_cosine_kernel(self):
        X = np.random.randn(4, 3, 50, 100)
        Y = np.random.randn(4, 3, 100, 100)
        ref = np.stack([
            np.stack([
                ck_ref(x) for x in X_
            ]) for X_ in X])
        out = cosine_kernel(X)
        assert np.allclose(out, ref, atol=1e-5)
        ref = np.stack([
            np.stack([
                ck_ref(x, y) for x, y in zip(X_, Y_)
            ]) for X_, Y_ in zip(X, Y)])
        out = cosine_kernel(X, Y)
        assert np.allclose(out, ref, atol=1e-5)

        key = jax.random.PRNGKey(4839)
        k0, k1 = jax.random.split(key, 4)
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
        ref = np.stack([
            np.stack([
                ck_ref(x) for x in X_
            ]) for X_ in X.todense()])
        out = cosine_kernel(X)
        assert np.allclose(out, ref, atol=1e-5)
        ref = np.stack([
            np.stack([
                ck_ref(x, y) for x, y in zip(X_, Y_)
            ]) for X_, Y_ in zip(X.todense(), Y.todense())])
        out = cosine_kernel(X, Y)
        assert np.allclose(out, ref, atol=1e-5)

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
        k0, k1, k2, k3 = jax.random.split(key, 4)
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

        theta = jax.random.normal(k2, shape=(3, 100, 100))
        out = linear_kernel(X, Y, theta=theta)
        ref = linear_kernel(X.todense(), Y.todense(), theta=theta)
        assert np.allclose(out, ref, atol=1e-5)

        #TODO: We're not doing any kind of correctness checks here.
        k = 4
        ikeys = jax.random.split(k3, 50)
        indices_L = jnp.stack([
            jax.random.choice(i, a=100, shape=(k, 1), replace=False)
            for i in ikeys
        ], axis=0)
        ikeys = jax.random.split(k3, 100)
        indices_R = jnp.stack([
            jax.random.choice(i, a=100, shape=(k, 1), replace=False)
            for i in ikeys
        ], axis=0)
        theta = random_sparse(
            (100, 100),
            k=5,
            key=k2
        )
        linear_kernel(X, Y, theta=theta,
                      intermediate_indices=(indices_L, indices_R))
        theta = jax.random.normal(k2, shape=(3, 100, 100))
        linear_kernel(X, Y, theta=theta, intermediate_indices=indices_R)

    def test_linear_distance(self):
        X = np.random.randn(4, 3, 50, 100)
        Y = np.random.randn(4, 3, 100, 100)
        out = jnp.sqrt(linear_distance(X, Y))
        ref = np.stack([
            np.stack([
                cdist(x, y) for x, y in zip(X_, Y_)
            ]) for X_, Y_ in zip(X, Y)])
        assert np.allclose(out, ref, atol=1e-5)

        theta = np.random.rand(100,)
        out = jnp.sqrt(linear_distance(X, Y, theta=theta))
        ref = np.stack([
            np.stack([
                cdist(x, y, metric='mahalanobis', VI=np.diagflat(theta))
                for x, y in zip(X_, Y_)
            ]) for X_, Y_ in zip(X, Y)])
        assert np.allclose(out, ref, atol=1e-5)

        theta = np.random.rand(3, 100,)
        out = jnp.sqrt(linear_distance(X, Y, theta=theta))
        ref = np.stack([
            np.stack([
                cdist(x, y, metric='mahalanobis', VI=np.diagflat(th))
                for x, y, th in zip(X_, Y_, theta)
            ]) for X_, Y_ in zip(X, Y)])
        assert np.allclose(out, ref, atol=1e-5)

        key = jax.random.PRNGKey(4839)
        k0, k1, k2 = jax.random.split(key, 4)
        X = random_sparse(
            (4, 3, 50, 100),
            k=5,
            key=k0
        )
        Y = random_sparse(
            (4, 3, 70, 100),
            k=5,
            key=k1
        )
        out = jnp.sqrt(linear_distance(X, Y))
        ref = np.stack([
            np.stack([
                cdist(x, y) for x, y in zip(X_, Y_)
            ]) for X_, Y_ in zip(X.todense(), Y.todense())])
        assert np.allclose(out, ref, atol=1e-5)

        theta = np.random.rand(100,)
        out = jnp.sqrt(linear_distance(X, Y, theta=theta))
        ref = np.stack([
            np.stack([
                cdist(x, y, metric='mahalanobis', VI=np.diagflat(theta))
                for x, y in zip(X_, Y_)
            ]) for X_, Y_ in zip(X.todense(), Y.todense())])
        assert np.allclose(out, ref, atol=1e-5)

        theta = np.random.rand(3, 100,)
        out = jnp.sqrt(linear_distance(X, Y, theta=theta))
        ref = np.stack([
            np.stack([
                cdist(x, y, metric='mahalanobis', VI=np.diagflat(th))
                for x, y, th in zip(X_, Y_, theta)
            ]) for X_, Y_ in zip(X.todense(), Y.todense())])
        assert np.allclose(out, ref, atol=1e-5)

        theta = random_sparse(
            (100, 100),
            k=5,
            key=k2
        )
        out = jnp.sqrt(linear_distance(X, Y, theta=theta))
        theta_ref = theta.todense().T @ theta.todense()
        ref = np.stack([
            np.stack([
                cdist(x, y, metric='mahalanobis', VI=theta_ref)
                for x, y in zip(X_, Y_)
            ]) for X_, Y_ in zip(X.todense(), Y.todense())])
        assert np.allclose(out, ref, atol=1e-5)

        theta = jax.random.normal(k2, shape=(3, 100, 100))
        theta = theta @ theta.swapaxes(-1, -2) # make sure it's positive definite
        out = jnp.sqrt(linear_distance(X, Y, theta=theta))
        ref = np.stack([
            np.stack([
                cdist(x, y, metric='mahalanobis', VI=th)
                for x, y, th in zip(X_, Y_, theta)
            ]) for X_, Y_ in zip(X.todense(), Y.todense())])
        assert np.allclose(out, ref, atol=1e-5)
