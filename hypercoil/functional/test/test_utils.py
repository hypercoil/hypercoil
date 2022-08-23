# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for utility functions.
"""
import jax
import jax.numpy as jnp
import numpy as np
from hypercoil.functional.utils import (
    apply_mask, wmean, orient_and_conform,
    conform_mask, mask_tensor, vmap_over_outer,
    promote_axis, demote_axis, fold_axis, unfold_axes,
    axis_complement, standard_axis_number,
    fold_and_promote, demote_and_unfold
)


class TestUtils:

    def test_wmean(self):
        z = np.array([[
            [1., 4., 2.],
            [0., 9., 1.],
            [4., 6., 7.]],[
            [0., 9., 1.],
            [4., 6., 7.],
            [1., 4., 2.]
        ]])
        w = np.ones_like(z)
        assert np.allclose(wmean(z, w), jnp.mean(z))
        w = np.array([1., 0., 1.])
        assert np.all(wmean(z, w, axis=1) == np.array([
            [(1 + 4) / 2, (4 + 6) / 2, (2 + 7) / 2],
            [(0 + 1) / 2, (9 + 4) / 2, (1 + 2) / 2]
        ]))
        assert np.all(wmean(z, w, axis=2) == np.array([
            [(1 + 2) / 2, (0 + 1) / 2, (4 + 7) / 2],
            [(0 + 1) / 2, (4 + 7) / 2, (1 + 2) / 2]
        ]))
        w = np.array([
            [1., 0., 1.],
            [0., 1., 1.]
        ])
        assert np.all(wmean(z, w, axis=(0, 1)) == np.array([
            [(1 + 4 + 4 + 1) / 4, (4 + 6 + 6 + 4) / 4, (2 + 7 + 7 + 2) / 4]
        ]))
        assert np.all(wmean(z, w, axis=(0, 2)) == np.array([
            [(1 + 2 + 9 + 1) / 4, (0 + 1 + 6 + 7) / 4, (4 + 7 + 4 + 2) / 4]
        ]))

    def test_mask(self):
        msk = jnp.array([1, 1, 0, 0, 0], dtype=bool)
        tsr = np.random.rand(5, 5, 5)
        mskd = apply_mask(tsr, msk, axis=0)
        assert mskd.shape == (2, 5, 5)
        assert np.all(mskd == tsr[:2])
        mskd = apply_mask(tsr, msk, axis=1)
        assert mskd.shape == (5, 2, 5)
        assert np.all(mskd == tsr[:, :2])
        mskd = apply_mask(tsr, msk, axis=2)
        assert np.all(mskd == tsr[:, :, :2])
        assert mskd.shape == (5, 5, 2)
        mskd = apply_mask(tsr, msk, axis=-1)
        assert np.all(mskd == tsr[:, :, :2])
        assert mskd.shape == (5, 5, 2)
        mskd = apply_mask(tsr, msk, axis=-2)
        assert np.all(mskd == tsr[:, :2])
        assert mskd.shape == (5, 2, 5)
        mskd = apply_mask(tsr, msk, axis=-3)
        assert np.all(mskd == tsr[:2])
        assert mskd.shape == (2, 5, 5)

        mask = conform_mask(tsr[0, 0], msk, axis=-1, batch=True)
        assert mask.shape == (5,)
        assert tsr[0, 0][mask].size == 2
        mask = conform_mask(tsr, msk, axis=-1)
        assert mask.shape == (5, 5, 5)
        assert tsr[mask].size == 50
        mask = conform_mask(tsr, jnp.outer(msk, msk), axis=-1, batch=True)
        assert mask.shape == (5, 5, 5)
        assert tsr[mask].size == 20

        jconform = jax.jit(conform_mask, static_argnames=('axis', 'batch'))
        mask = jconform(tsr[0, 0], msk, axis=-1, batch=True)
        assert mask.shape == (5,)
        assert tsr[0, 0][mask].size == 2
        mask = jconform(tsr, msk, axis=-1)
        assert mask.shape == (5, 5, 5)
        assert tsr[mask].size == 50
        mask = jconform(tsr, jnp.outer(msk, msk), axis=-1, batch=True)
        assert mask.shape == (5, 5, 5)
        assert tsr[mask].size == 20

        jtsrmsk = jax.jit(mask_tensor, static_argnames=('axis',))
        mskd = jtsrmsk(tsr, msk, axis=-1)
        assert (mskd != 0).sum() == 50
        mskd = jtsrmsk(tsr, msk, axis=-1, fill_value=float('nan'))
        assert np.isnan(mskd).sum() == 75

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

    def test_orient_and_conform(self):
        X = np.random.rand(3, 7)
        R = np.random.rand(2, 7, 11, 1, 3)
        out = orient_and_conform(X.swapaxes(-1, 0), (1, -1), reference=R)
        ref = X.swapaxes(-1, -2)[None, :, None, None, :]
        assert(out.shape == ref.shape)
        assert np.all(out == ref)

        X = np.random.rand(7)
        R = np.random.rand(2, 7, 11, 1, 3)
        out = orient_and_conform(X, 1, reference=R)
        ref = X[None, :, None, None, None]
        assert(out.shape == ref.shape)
        assert np.all(out == ref)

        # test with jit compilation
        jorient = jax.jit(orient_and_conform, static_argnames=('axis', 'dim'))
        out = jorient(X, 1, dim=R.ndim)
        ref = X[None, :, None, None, None]
        assert(out.shape == ref.shape)
        assert np.all(out == ref)
