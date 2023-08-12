# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for residualisation
"""
import jax
import jax.numpy as jnp
from hypercoil.functional.resid import residualise


class TestResidualisation:
    def test_residualisation(self):
        key = jax.random.PRNGKey(0)
        xkey, ykey = jax.random.split(key)
        X = jax.random.normal(key=xkey, shape=(3, 30, 100))
        Y = jax.random.normal(key=ykey, shape=(3, 1000, 100))
        out = residualise(Y, X)
        assert out.shape == (3, 1000, 100)

        X = jnp.ones((3, 100, 1))
        Y = jax.random.normal(key=ykey, shape=(3, 100, 1000))
        out = residualise(Y, X, rowvar=False)
        assert out.shape == (3, 100, 1000)
        assert jnp.allclose(out.mean(-2), 0, atol=1e-5)

        X = jax.random.normal(key=xkey, shape=(3, 30, 100))
        Y = jax.random.normal(key=ykey, shape=(3, 1000, 100))
        out = residualise(Y, X, l2=0.01)
        assert not jnp.allclose(out, Y)
        out = residualise(Y, X, l2=10000.)
        assert jnp.allclose(out, Y, atol=1e-5)
        out = residualise(Y, X, l2=10000., return_mode='orthogonal')
        assert jnp.allclose(out, 0, atol=1e-5)
