# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for activation function modules
"""
import jax
from hypercoil.nn.activation import CorrelationNorm, Isochor


class TestActivationModules:
    def test_corrnorm(self):
        key = jax.random.PRNGKey(0)
        X = jax.random.uniform(key=key, shape=(4, 2, 50, 50))
        out = jax.jit(CorrelationNorm())(X)
        assert out.shape == X.shape

    def test_isochor(self):
        key = jax.random.PRNGKey(0)
        X = jax.random.uniform(key=key, shape=(4, 2, 50, 50))
        X = X @ X.swapaxes(-1, -2)
        out = jax.jit(Isochor())(X)
        assert out.shape == X.shape
