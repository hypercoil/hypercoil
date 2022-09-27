# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for covariance modules
"""
import pytest
import numpy as np
import jax
import distrax
import equinox as eqx
from hypercoil.engine.noise import (
    StochasticParameter,
    Diagonal, Symmetric, MatrixExponential,
    TensorIIDAddStochasticTransform,
    TensorIIDMulStochasticTransform,
    ScalarIIDAddStochasticTransform,
    ScalarIIDMulStochasticTransform,
)
from hypercoil.functional.cov import (
    corr, pcorr, pairedcorr
)
from hypercoil.functional.utils import apply_mask
from hypercoil.nn import (
    UnaryCovariance, UnaryCovarianceTW, UnaryCovarianceUW,
    BinaryCovariance, BinaryCovarianceTW, BinaryCovarianceUW
)


class TestCovNN:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        import os
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.approx = lambda x, y: np.allclose(x, y, atol=1e-5)
        self.approxh = lambda x, y: np.allclose(x, y, atol=1e-3)

        key = jax.random.PRNGKey(0)
        xkey, ykey, srckey = jax.random.split(key, 3)

        self.n = 100
        self.X = jax.random.uniform(key=xkey, shape=(4, 13, self.n))
        self.Y = jax.random.uniform(key=ykey, shape=(4, 7, self.n))

        inner_distr = distrax.Normal(3, 1)
        distr = Diagonal(
            src_distribution=inner_distr,
            multiplicity=self.n,
        )
        self.nns = ScalarIIDAddStochasticTransform(
            distribution=inner_distr, key=srckey)
        self.dns = TensorIIDAddStochasticTransform(
            distribution=distr,
            event_axes=(-1, -2),
            key=srckey
        )

        inner_distr = distrax.Bernoulli(probs=0.3)
        self.bds = ScalarIIDMulStochasticTransform(
            distribution=inner_distr, key=srckey)
        distr = Diagonal(
            src_distribution=inner_distr,
            multiplicity=self.n,
        )
        self.dds = TensorIIDMulStochasticTransform(
            distribution=distr,
            event_axes=(-1, -2),
            key=srckey
        )

        distr = Symmetric(
            src_distribution=inner_distr,
            multiplicity=self.n
        )
        distr = MatrixExponential(distr)
        self.psds = TensorIIDMulStochasticTransform(
            distribution=distr,
            event_axes=(-1, -2),
            key=srckey
        )

    def test_cov_uuw(self):
        key = jax.random.PRNGKey(0)
        mkey, skey = jax.random.split(key, 2)
        cov = UnaryCovarianceUW(
            estimator=corr,
            dim=self.n,
            key=mkey,
        )
        out = eqx.filter_jit(cov)(self.X)
        ref = np.stack([np.corrcoef(x) for x in self.X])
        assert self.approx(out, ref)
        cov = UnaryCovarianceUW(
            estimator=pcorr,
            dim=self.n,
            out_channels=7,
            key=mkey,
        )
        weight = self.dns.sample(
            key=skey,
            shape=(cov.out_channels, cov.dim, cov.dim)
        )
        eqx.filter_jit(cov)(self.X, weight=weight)

    def test_cov_uuw_masked(self):
        key = jax.random.PRNGKey(0)
        mkey, skey = jax.random.split(key, 2)
        cov = UnaryCovarianceUW(
            estimator=corr,
            dim=self.n,
            key=mkey,
        )
        mask = jax.random.bernoulli(key=skey, shape=(self.n,))
        out = eqx.filter_jit(cov)(self.X, mask=mask)
        ref = np.stack([
            np.corrcoef(apply_mask(x, mask, -1))
            for x in self.X
        ])
        assert self.approx(out, ref)
        mask = jax.random.bernoulli(key=skey, shape=(4, 1, self.n))
        out = eqx.filter_jit(cov)(self.X, mask=mask)
        ref = np.stack([
            np.corrcoef(apply_mask(x, m, -1))
            for x, m in zip(self.X, mask)
        ])
        assert self.approx(out, ref)

    def test_cov_utw(self):
        cov = UnaryCovarianceTW(
            estimator=corr,
            dim=self.n,
        )
        out = eqx.filter_jit(cov)(self.X)
        ref = np.stack([np.corrcoef(x) for x in self.X])
        assert self.approx(out, ref)

        cov = UnaryCovarianceTW(
            estimator=pcorr,
            dim=self.n,
            max_lag=3,
            out_channels=7,
        )
        assert cov.weight_row.shape == (7, 4)
        assert cov.weight[5, 15, 17] == cov.weight[5, 94, 96]
        assert cov.weight[3, 13, 17] == 0
        eqx.filter_jit(cov)(self.X)
        cov = StochasticParameter.wrap(
            cov, where='weight_col;weight_row', transform=self.nns
        )
        cov = StochasticParameter.wrap(
            cov, where='weight_col;weight_row', transform=self.bds
        )
        eqx.filter_jit(cov)(self.X)

    def test_cov_uw(self):
        cov = UnaryCovariance(
            estimator=corr,
            dim=self.n,
        )
        out = eqx.filter_jit(cov)(self.X)
        ref = np.stack([np.corrcoef(x) for x in self.X])
        assert self.approx(out, ref)

        cov = UnaryCovariance(
            estimator=pcorr,
            dim=self.n,
            min_lag=1,
            max_lag=3,
            out_channels=7,
        )
        eqx.filter_jit(cov)(self.X)

        cov = UnaryCovariance(
            estimator=corr,
            dim=self.n,
            min_lag=None,
            max_lag=None,
        )
        eqx.filter_jit(cov)(self.X)

    def test_cov_buw(self):
        cov = BinaryCovarianceUW(
            estimator=pairedcorr,
            dim=self.n,
        )
        out = eqx.filter_jit(cov)(self.X, self.Y)
        ref = np.stack([np.corrcoef(x, y)[:13, -7:]
                        for x, y in zip(self.X, self.Y)])
        assert self.approx(out, ref)

    def test_cov_btw(self):
        cov = BinaryCovarianceTW(
            estimator=pairedcorr,
            dim=self.n,
        )
        out = eqx.filter_jit(cov)(self.X, self.Y)
        ref = np.stack([np.corrcoef(x, y)[:13, -7:]
                        for x, y in zip(self.X, self.Y)])
        assert self.approx(out, ref)

        cov = BinaryCovarianceTW(
            estimator=pairedcorr,
            dim=self.n,
            max_lag=-1,
            min_lag=-3,
            out_channels=2,
        )
        eqx.filter_jit(cov)(self.X, self.Y)

    def test_cov_bw(self):
        cov = BinaryCovariance(
            estimator=pairedcorr,
            dim=self.n,
        )
        out = eqx.filter_jit(cov)(self.X, self.Y)
        ref = np.stack([np.corrcoef(x, y)[:13, -7:]
                        for x, y in zip(self.X, self.Y)])
        assert self.approx(out, ref)
