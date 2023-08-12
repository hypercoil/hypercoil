# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for residualisation module
"""
import jax
import jax.numpy as jnp
from hypercoil.nn.resid import Residualise


class TestResidualise:

    def test_residualise(self):
        key = jax.random.PRNGKey(0)
        xkey, ykey = jax.random.split(key)
        Y = jax.random.normal(key=xkey, shape=(2, 2, 100, 5))
        X = jax.random.normal(key=ykey, shape=(2, 1, 1, 5))
        out = jax.jit(Residualise())(Y, X)
        assert out.shape == Y.shape

        Y = jax.random.normal(key=xkey, shape=(2, 1, 100, 5))
        X = jax.random.normal(key=ykey, shape=(2, 2, 1, 5))
        out = jax.jit(Residualise())(Y, X)
        assert out.shape == (2, 2, 100, 5) # X doubles the number of channels
