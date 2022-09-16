# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for loss functions.
"""
import jax
import jax.numpy as jnp
from hypercoil.loss.functional import identity
from hypercoil.loss.scalarise import (
    selfwmean, wmean, wmean_scalarise, selfwmean_scalarise,
)


class TestLossFunction:

    def test_wmean(self):
        z = jnp.array([[
            [1., 4., 2.],
            [0., 9., 1.],
            [4., 6., 7.]],[
            [0., 9., 1.],
            [4., 6., 7.],
            [1., 4., 2.]
        ]])
        w = jnp.ones_like(z)
        assert jnp.allclose(wmean(z, w), jnp.mean(z))
        w = jnp.array([1., 0., 1.])
        assert jnp.all(wmean(z, w, axis=1) == jnp.array([
            [(1 + 4) / 2, (4 + 6) / 2, (2 + 7) / 2],
            [(0 + 1) / 2, (9 + 4) / 2, (1 + 2) / 2]
        ]))
        assert jnp.all(wmean(z, w, axis=2) == jnp.array([
            [(1 + 2) / 2, (0 + 1) / 2, (4 + 7) / 2],
            [(0 + 1) / 2, (4 + 7) / 2, (1 + 2) / 2]
        ]))
        w = jnp.array([
            [1., 0., 1.],
            [0., 1., 1.]
        ])
        assert jnp.all(wmean(z, w, axis=(0, 1)) == jnp.array([
            [(1 + 4 + 4 + 1) / 4, (4 + 6 + 6 + 4) / 4, (2 + 7 + 7 + 2) / 4]
        ]))
        assert jnp.all(wmean(z, w, axis=(0, 2)) == jnp.array([
            [(1 + 2 + 9 + 1) / 4, (0 + 1 + 6 + 7) / 4, (4 + 7 + 4 + 2) / 4]
        ]))

        loss = jax.jit(wmean_scalarise(identity, axis=(0, 1)))
        out = loss(z, scalarisation_weight=w)
        assert jnp.all(out == jnp.array([
            [(1 + 4 + 4 + 1) / 4, (4 + 6 + 6 + 4) / 4, (2 + 7 + 7 + 2) / 4]
        ]).mean())

    def test_selfwmean(self):
        key = jax.random.PRNGKey(0)
        X = jnp.array([
            [-100, -100, 0, -100, -100],
            [0, -100, -100, -100, -100],
            [-100, -100, -100, -100., 0]
        ])
        Y = jax.random.normal(key=key, shape=(3, 5))

        assert jnp.isclose(selfwmean(X, softmax_axis=-1), 0)
        assert not jnp.isclose(selfwmean(Y, softmax_axis=-1), 0)

        loss = jax.jit(selfwmean_scalarise(identity, axis=None, softmax_axis=-1))
        assert jnp.isclose(loss(X), 0)
