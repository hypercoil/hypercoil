# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for symmetric matrix maps
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
from scipy.linalg import expm, logm, sqrtm, sinm, funm
from hypercoil.functional import (
    symmap, symexp, symlog, symsqrt
)


class TestSymmetricMap:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.tol = 1e-5
        self.rtol = 1e-4
        self.approx = lambda out, ref: np.allclose(
            out, ref, atol=self.tol, rtol=self.rtol)
        np.random.seed(10)

        A = np.random.rand(10, 10)
        B = np.random.rand(4, 100, 5) # low-rank, singular!
        AM = np.random.rand(200, 10, 10)
        AS = np.random.rand(200, 10, 4)
        self.A = A @ A.T
        self.B = B = B @ B.swapaxes(-1, -2)
        self.AM = AM @ np.swapaxes(AM, -1, -2)
        self.AS = AS @ np.swapaxes(AS, -1, -2)

    def test_expm(self):
        out = symexp(self.A)
        ref = expm(self.A)
        assert self.approx(out, ref)

    def test_logm(self):
        out = symlog(self.A)
        ref = logm(self.A)
        # Note that this is a very weak condition! More optimistically,
        # only a few entries have errors larger than 1e-3.
        assert np.allclose(out, ref, atol=1e-2, rtol=1e-2)

    def test_sqrtm(self):
        out = symsqrt(self.A)
        ref = sqrtm(self.A)
        assert np.allclose(out, ref, atol=1e-3, rtol=1e-3)

    def test_map(self):
        out = symmap(self.B, lambda x: x, truncate_eigenvalues=True)
        assert np.allclose(out, self.B)

        out = symmap(self.A, jnp.sin)
        ref = funm(self.A, np.sin)
        assert self.approx(out, ref)
        ref = sinm(self.A)
        assert self.approx(out, ref)

    def test_map_multidim(self):
        out = symmap(self.AM, jnp.exp)
        ref = np.stack([expm(AMi) for AMi in self.AM])
        assert self.approx(out, ref)

    def test_singular(self):
        key = jax.random.PRNGKey(0)
        out = symmap(self.AS, jnp.log, fill_nans=False)
        assert np.all(np.logical_or(np.isnan(out), np.isinf(out)))
        out = symmap(self.AS, jnp.log, psi=1e-3, key=key)
        assert np.all(np.logical_not(np.logical_or(
            np.isnan(out), np.isinf(out))))
