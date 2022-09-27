# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for interpolation modules
"""
import jax
import jax.numpy as jnp
from hypercoil.nn.interpolate import (
    SpectralInterpolate,
    LinearInterpolate,
    HybridInterpolate,
)


# These are absolutely minimal tests. The actual functionality is tested in
# ``hypercoil.functional.interpolate`` since these modules are just thin
# wrappers around those functions.
class TestInterpolate:
    def test_spectral(self):
        key = jax.random.PRNGKey(0)
        dkey, mkey = jax.random.split(key)
        X = jax.random.normal(dkey, (2, 1, 3, 200))
        mask = jax.random.bernoulli(mkey, 0.5, (2, 1, 1, 200))
        jax.jit(SpectralInterpolate())(X, mask)

    def test_linear(self):
        key = jax.random.PRNGKey(0)
        dkey, mkey = jax.random.split(key)
        X = jax.random.normal(dkey, (2, 1, 3, 200))
        mask = jax.random.bernoulli(mkey, 0.2, (2, 1, 1, 200))
        out = jax.jit(LinearInterpolate())(X, mask)
        pts = jnp.arange(200)
        for i, (x, msk) in enumerate(zip(X, mask)):
            msk = msk.squeeze()
            for j in range(3):
                data = x[0, j, :]
                r = jnp.interp(pts, jnp.where(msk)[0], data[jnp.where(msk)])
                o = out[i, 0, j, :]
                assert jnp.allclose(r, o, atol=1e-3)

    def test_hybrid(self):
        key = jax.random.PRNGKey(0)
        dkey, mkey = jax.random.split(key)
        X = jax.random.normal(dkey, (2, 1, 3, 200))
        mask = jax.random.bernoulli(mkey, 0.5, (2, 1, 1, 200))
        jax.jit(HybridInterpolate())(X, mask)
