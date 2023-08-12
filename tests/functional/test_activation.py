# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for activation functions
"""
import jax
import jax.numpy as jnp
import numpy as np
from hypercoil.functional import cov, corr
from hypercoil.functional.activation import (
    laplace, expbarrier,
    amplitude_tanh, amplitude_atanh,
    amplitude_laplace, amplitude_expbarrier,
    corrnorm, isochor
)


class TestActivationFunctions:

    def test_activations_forward(self):
        A = np.random.rand(5, 5, 5)
        laplace(A)
        expbarrier(A)
        amplitude_tanh(A)
        amplitude_atanh(A)
        amplitude_laplace(A)
        amplitude_expbarrier(A)
        jax.jit(laplace)(A)
        jax.jit(expbarrier)(A)
        jax.jit(amplitude_tanh)(A)
        jax.jit(amplitude_atanh)(A)
        jax.jit(amplitude_laplace)(A)
        jax.jit(amplitude_expbarrier)(A)

        assert np.allclose(amplitude_tanh(amplitude_atanh(A)), A)

    def test_corrnorm(self):
        A = np.random.rand(12, 30)
        assert np.all(corrnorm(cov(A)) == corr(A))
        B = np.random.randn(12, 12)
        B = B + B.T
        assert np.all(
            jnp.sign(jnp.diagonal(B)) ==
            jnp.sign(jnp.diagonal(corrnorm(B)))
        )

        # test jit
        jit_corrnorm = jax.jit(corrnorm)
        assert np.all(jit_corrnorm(cov(A)) == corr(A))

    def test_isochor(self):
        # on rare occasions it's outside tolerance
        key = jax.random.PRNGKey(238)
        subkey, key = jax.random.split(key)
        A = jax.random.normal(subkey, (5, 20, 20))
        A = A @ A.swapaxes(-1, -2)
        out = isochor(A)
        assert np.allclose(jnp.linalg.det(out), 1, atol=1e-2)

        out = isochor(A, volume=4)
        assert np.allclose(jnp.linalg.det(out), 4, atol=1e-2)

        out = isochor(A, volume=4, max_condition=5)
        assert np.allclose(jnp.linalg.det(out), 4, atol=1e-2)
        L, Q = jnp.linalg.eigh(out)
        assert np.all((L.max(-1) / L.min(-1)) < 5.01)

        out = isochor(A, softmax_temp=1e10)
        assert np.allclose(out, jnp.eye(20), atol=1e-4)

        out = isochor(A, volume=(2 ** 20), max_condition=1)
        assert np.allclose(out, 2 * jnp.eye(20), atol=1e-4)

        # test jit
        jit_isochor = jax.jit(
            isochor,
            static_argnames=('volume', 'max_condition', 'softmax_temp'))
        out = jit_isochor(A, volume=4, max_condition=5)
        assert np.allclose(jnp.linalg.det(out), 4, atol=1e-2)
        L, Q = jnp.linalg.eigh(out)
        assert np.all((L.max(-1) / L.min(-1)) < 5.01)

        out = jit_isochor(A, softmax_temp=1e10)
        assert np.allclose(out, jnp.eye(20), atol=1e-4)
