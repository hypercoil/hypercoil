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
    conform_mask, mask_tensor,
)
#TODO: Move these tests!
from distrax import MultivariateNormalFullCovariance
from hypercoil.engine import (
    sample_multivariate,
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
        tsr = jnp.asarray(tsr)
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

    def test_multivariate_sample(self):
        mu = np.array([100, 0, -100])
        sigma = np.random.randn(3, 3)
        sigma = sigma @ sigma.T
        distr = MultivariateNormalFullCovariance(mu, sigma)
        out = sample_multivariate(
            distr=distr,
            shape=(2, 3, 100),
            event_axes=(-2,),
            key=jax.random.PRNGKey(0),
        )
        assert out.shape == (2, 3, 100)
        assert np.all(np.abs(out.mean((0, -1)) - mu) < 0.5)

        mu = np.array([100, 100, 100])
        distr = MultivariateNormalFullCovariance(mu, sigma)
        out = sample_multivariate(
            distr=distr,
            shape=(2, 3, 100),
            event_axes=(-2,),
            mean_correction=True,
            key=jax.random.PRNGKey(0),
        )
        assert np.abs(out.mean() - 1) < 0.1
        assert np.abs(out.mean() - 100) > 90
