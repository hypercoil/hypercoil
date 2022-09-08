# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for axis utility functions.
"""
import jax
import jax.numpy as jnp
import numpy as np
from hypercoil.engine.axisutil import (
    vmap_over_outer, broadcast_ignoring,
    promote_axis, demote_axis, fold_axis, unfold_axes,
    axis_complement, standard_axis_number,
    fold_and_promote, demote_and_unfold,
)


class TestAxisUtil:

    def test_vmap_over_outer(self):
        test_obs = 100
        offset = 10
        offset2 = 50
        w = np.zeros((test_obs, test_obs))
        rows, cols = np.diag_indices_from(w)
        w[(rows, cols)] = np.random.rand(test_obs)
        w[(rows[:-offset], cols[offset:])] = (
            3 * np.random.rand(test_obs - offset))
        w[(rows[:-offset2], cols[offset2:])] = (
            np.random.rand(test_obs - offset2))
        w = jnp.stack([w] * 20)
        w = w.reshape(2, 2, 5, test_obs, test_obs)

        jaxpr_test = jax.make_jaxpr(vmap_over_outer(jnp.diagonal, 2))((w,))
        jaxpr_ref = jax.make_jaxpr(
            jax.vmap(jax.vmap(jax.vmap(jnp.diagonal, 0, 0), 1, 1), 2, 2))(w)
        assert jaxpr_test.jaxpr.pretty_print() == jaxpr_ref.jaxpr.pretty_print()

        out = vmap_over_outer(jnp.diagonal, 2)((w,))
        ref = jax.vmap(jax.vmap(jax.vmap(jnp.diagonal, 0, 0), 1, 1), 2, 2)(w)
        assert np.allclose(out, ref)

        out = jax.jit(vmap_over_outer(jnp.diagonal, 2))((w,))
        ref = jax.jit(jax.vmap(jax.vmap(jax.vmap(jnp.diagonal, 0, 0), 1, 1), 2, 2))(w)
        assert np.allclose(out, ref)

        L = np.random.rand(5, 13)
        R = np.random.rand(2, 5, 4)
        jvouter = jax.jit(vmap_over_outer(jnp.outer, 1))
        out = jvouter((L, R))
        ref = jax.vmap(jax.vmap(jnp.outer, (None, 0), 0), (0, 1), 1)(L, R)
        assert out.shape == (2, 5, 13, 4)
        assert np.allclose(out, ref)

    def test_axis_ops(self):
        shape = (2, 3, 5, 7, 11)
        X = np.empty(shape)
        ndim = X.ndim
        assert axis_complement(ndim, -2) == (0, 1, 2, 4)
        assert axis_complement(ndim, (0, 1, 4)) == (2, 3)
        assert axis_complement(ndim, (0, 1, 2, 3, -1)) == ()

        assert standard_axis_number(-2, ndim) == 3
        assert standard_axis_number(1, ndim) == 1

        assert unfold_axes(X, (-3, -2)).shape == (2, 3, 35, 11)
        assert unfold_axes(X, (1, 2, 3)).shape == (2, 105, 11)

        assert promote_axis(ndim, -2) == (3, 0, 1, 2, 4)
        assert promote_axis(ndim, 1) == (1, 0, 2, 3, 4)

        assert fold_axis(X, -3, 1).shape == (2, 3, 5, 1, 7, 11)
        assert fold_axis(X, -3, 5).shape == (2, 3, 1, 5, 7, 11)

        assert demote_axis(7, (5, 2)) == (2, 3, 0, 4, 5, 1, 6)
        assert demote_axis(ndim, 2) == (1, 2, 0, 3, 4)

        assert fold_and_promote(X, -2, 7).shape == (7, 2, 3, 5, 1, 11)
        assert fold_and_promote(X, -4, 3).shape == (3, 2, 1, 5, 7, 11)

        assert demote_and_unfold(X, -2, (3, 4)).shape == (3, 5, 7, 22)
        assert demote_and_unfold(X, 1, (1, 2, 3)).shape == (3, 70, 11)

        X2 = np.random.rand(4, 3, 100, 7)
        Y = fold_and_promote(X2, -2, 5)
        assert Y.shape == (5, 4, 3, 20, 7)
        X2_hat = demote_and_unfold(Y, -2, (-3, -2))
        assert np.all(X2 == X2_hat)

        Y = demote_and_unfold(X2, -2, (-3, -2))
        assert Y.shape == (3, 400, 7)
        X2_hat = fold_and_promote(Y, -2, 4)
        assert np.all(X2 == X2_hat)

    def test_broadcast_ignoring(self):
        shapes = (
            (
                ((2, 3, 2), (4, 2)),
                ((2, 3, 2), (2, 4, 2))
            ),
            (
                ((2, 3, 2), (2,)),
                ((2, 3, 2), (2, 1, 2))
            ),
        )
        for ((x_in, y_in), (x_out, y_out)) in shapes:
            X, Y = broadcast_ignoring(
                jnp.zeros(x_in),
                jnp.zeros(y_in), axis=-2
            )
            assert X.shape == x_out
            assert Y.shape == y_out
