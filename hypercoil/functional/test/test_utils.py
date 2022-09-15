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
    apply_mask, conform_mask, mask_tensor,
)
#TODO: Move these tests!
from distrax import MultivariateNormalFullCovariance
from hypercoil.engine import (
    sample_multivariate,
)


class TestUtils:

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
