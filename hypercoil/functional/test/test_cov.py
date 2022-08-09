# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for covariance, correlation, and derived measures
"""
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import pingouin # required for pcorr test
import pandas as pd
from hypercoil.functional.cov import (
    cov, corr, partialcorr, pairedcov, pairedcorr, precision,
    conditionalcov, conditionalcorr,
    _prepare_weight_and_avg, _prepare_denomfact
)


#TODO: Unit tests still needed for:
# - correctness of off-diagonal weighted covariance


class TestCov:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.tol = 5e-7
        self.ofunc = cov
        self.rfunc = np.cov
        self.approx = lambda out, ref: np.allclose(out, ref, atol=self.tol)

        self.x = np.random.rand(100)
        self.X = np.random.rand(7, 100)
        self.XM = np.random.rand(200, 7, 100)
        self.w = np.random.rand(100)
        self.wM = np.random.rand(3, 100)
        self.wMe = jax.vmap(jnp.diagflat, 0, 0)(self.wM)
        self.W = np.random.rand(100, 100)
        self.Y = np.random.rand(3, 100)

    def test_normalisations(self):
        test_obs = 2000
        test_dim = 10
        offset = 1000
        offset2 = 1500
        bias = False
        ddof = None
        # variances should all be close to 1
        X = np.random.randn(test_dim, test_obs)
        X = (X - X.mean(-1, keepdims=True)) / X.std(-1, keepdims=True)

        # No weights
        w = None
        weight, w_type, w_sum, (avg,) = _prepare_weight_and_avg((X,), w)
        fact = _prepare_denomfact(w_sum, w_type, ddof, bias, weight)
        assert(
            np.allclose(np.diag(X @ X.T / fact).mean(), 1, atol=0.1))

        # Vector weights
        w = np.random.rand(test_obs)
        weight, w_type, w_sum, (avg,) = _prepare_weight_and_avg((X,), w)
        fact = _prepare_denomfact(w_sum, w_type, ddof, bias, weight)
        assert(
            np.allclose(np.diag(X * weight @ X.T / fact).mean(), 1, atol=0.1))

        # Diagonal weights
        w = np.diag(np.random.rand(test_obs))
        weight, w_type, w_sum, (avg,) = _prepare_weight_and_avg((X,), w)
        fact = _prepare_denomfact(w_sum, w_type, ddof, bias, weight)
        assert(
            np.allclose(np.diag(X @ weight @ X.T / fact).mean(), 1, atol=0.1))

        # Nondiagonal weights
        #TODO: This is not working yet / actually not tested
        w = np.zeros((test_obs, test_obs))
        rows, cols = np.diag_indices_from(w)
        w[(rows, cols)] = np.random.rand(test_obs)
        w[(rows[:-offset], cols[offset:])] = (
            3 * np.random.rand(test_obs - offset))
        w[(rows[:-offset2], cols[offset2:])] = (
            np.random.rand(test_obs - offset2))
        weight, w_type, w_sum, (avg,) = _prepare_weight_and_avg((X,), w)
        fact = _prepare_denomfact(w_sum, w_type, ddof, bias, weight)
        out = np.diag(X @ weight @ X.T / fact).mean()

    def covpattern(self, **args):
        out = self.ofunc(self.X, **args)
        ref = self.rfunc(self.X, **args)
        assert self.approx(out, ref)

    def test_cov_vanilla(self):
        args = {}
        self.covpattern(**args)

    def test_cov_transpose(self):
        args = {'rowvar': False}
        self.covpattern(**args)

    def test_cov_biased(self):
        args = {'bias': True}
        self.covpattern(**args)

    def test_cov_custdof(self):
        args = {'ddof': 17}
        self.covpattern(**args)

    def test_cov_var(self):
        out = self.ofunc(self.x)
        ref = self.rfunc(self.x)
        assert self.approx(out, ref)

    def test_cov_weighted(self):
        out = self.ofunc(self.X, weight=self.w)
        ref = self.rfunc(self.X, aweights=self.w)
        assert self.approx(out, ref)

    def test_cov_multiweighted_1d(self):
        out = self.ofunc(self.X, weight=self.wMe)
        ref = np.stack([
            self.rfunc(self.X, aweights=self.wM[i, :])
            for i in range(self.wM.shape[0])
        ])
        assert self.approx(out, ref)

    def test_cov_multiweighted(self):
        out = cov(self.X, weight=self.wMe)
        ref = cov(self.X, weight=self.wMe[0, :])
        assert self.approx(out[0, :], ref)
        assert out.shape == (3, 7, 7)

    def test_cov_Weighted(self):
        out = self.ofunc(self.X, weight=self.W)

    def test_cov_multidim(self):
        out = self.ofunc(self.XM, weight=self.w)
        ref = np.stack([
            self.rfunc(self.XM[i, :, :].squeeze(), aweights=self.w)
            for i in range(self.XM.shape[0])
        ])
        assert self.approx(out, ref)

    def test_paired(self):
        out = pairedcov(self.X, self.Y)
        ref = np.cov(np.concatenate([self.X ,self.Y], -2))[:7, -3:]
        assert self.approx(out, ref)

    def test_corr(self):
        out = corr(self.X)
        ref = np.corrcoef(self.X)
        assert self.approx(out, ref)

    def test_pairedcorr(self):
        out = pairedcorr(self.X, self.Y)
        ref = corr(jnp.concatenate((self.X, self.Y)))[:7, 7:]
        assert self.approx(out, ref)

    def test_pcorr(self):
        out = partialcorr(self.X)
        ref = pd.DataFrame(self.X.T).pcorr().values
        assert self.approx(out, ref)

    def test_ccov(self):
        """
        Verify equivalence of the Schur complement approach and fit-based
        confound regression.
        """
        out = conditionalcov(self.X, self.Y)
        ref = jnp.linalg.pinv(
            precision(jnp.concatenate((self.X ,self.Y), -2))[:7, :7])
        assert self.approx(out, ref)
        Y_intercept = np.concatenate([self.Y, np.ones((1, 100))])
        ref = np.cov(
            self.X - np.linalg.lstsq(Y_intercept.T, self.X.T, rcond=None)[0].T
            @ Y_intercept)
        assert self.approx(out, ref)

    def test_ccorr(self):
        """
        Verify equivalence of the Schur complement approach and fit-based
        confound regression.
        """
        Y_intercept = np.concatenate([self.Y, np.ones((1, 100))])
        out = conditionalcorr(self.X, self.Y)
        ref = np.corrcoef(
            self.X - np.linalg.lstsq(Y_intercept.T, self.X.T, rcond=None)[0].T
            @ Y_intercept)
        assert self.approx(out, ref)
        out = jax.jit(conditionalcorr)(self.X, self.Y)
        assert self.approx(out, ref)
