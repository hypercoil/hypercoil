# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for covariance modules
"""
import pytest
import numpy as np
import torch, hypernova
from hypernova.nn import (
    UnaryCovariance, UnaryCovarianceTW, UnaryCovarianceUW,
    BinaryCovariance, BinaryCovarianceTW, BinaryCovarianceUW
)
from hypernova.init.base import DistributionInitialiser
from hypernova.init.laplace import LaplaceInit
from hypernova.functional.noise import (
    DiagonalNoiseSource, DiagonalDropoutSource, BandDropoutSource
)
from hypernova.functional.domain import Logit


class TestCovNN:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        import os
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.approx = lambda x, y: np.allclose(x.detach(), y, atol=1e-5)

        self.n = 100
        self.X = torch.rand(4, 13, self.n)
        self.Y = torch.rand(4, 7, self.n)
        self.dns = DiagonalNoiseSource()
        self.dds = DiagonalDropoutSource()
        self.bds = BandDropoutSource()

    def test_cov_uuw(self):
        cov = UnaryCovarianceUW(self.n, estimator=hypernova.functional.corr)
        out = cov(self.X)
        ref = np.stack([np.corrcoef(x) for x in self.X])
        assert self.approx(out, ref)
        cov = UnaryCovarianceUW(
            self.n, estimator=hypernova.functional.pcorr,
            out_channels=7, noise=self.dns) #, dropout=dds)
        cov(self.X)

    def test_cov_utw(self):
        cov = UnaryCovarianceTW(self.n, estimator=hypernova.functional.corr)
        out = cov(self.X)
        ref = np.stack([np.corrcoef(x) for x in self.X])
        assert self.approx(out, ref)
        init = LaplaceInit(
            loc=(0, 0), excl_axis=[1], domain=Logit(4)
        )
        cov = UnaryCovarianceTW(
            self.n, estimator=hypernova.functional.pcorr, max_lag=3,
            out_channels=7, noise=self.dns, dropout=self.bds, init=init)
        cov(self.X)
        assert cov.prepreweight_c.size() == torch.Size([4, 7])
        assert cov.weight[5, 15, 17] == cov.weight[5, 94, 96]
        assert cov.postweight[3, 13, 17] == 0

    def test_cov_uw(self):
        cov = UnaryCovariance(self.n, estimator=hypernova.functional.corr)
        out = cov(self.X)
        ref = np.stack([np.corrcoef(x) for x in self.X])
        assert self.approx(out, ref)

    def test_cov_uw_lag(self):
        #TODO: this only makes sure nothing crashes in the forward pass
        # not a test for correctness
        cov = UnaryCovariance(
            self.n,
            estimator=hypernova.functional.corr,
            max_lag=2
        )
        out = cov(self.X)

    def test_cov_uw_domain(self):
        #TODO: this only makes sure nothing crashes in the forward pass
        # not a test for correctness
        init = DistributionInitialiser(
            distr=torch.distributions.Normal(0.5, 0.02),
            domain=Logit()
        )
        cov = UnaryCovariance(
            self.n,
            estimator=hypernova.functional.cov,
            init=init
        )
        out = cov(self.X)

    def test_cov_buw(self):
        cov = BinaryCovarianceUW(
            self.n, estimator=hypernova.functional.pairedcorr)
        out = cov(self.X, self.Y)
        ref = np.stack([np.corrcoef(x, y)[:13, -7:]
                        for x, y in zip(self.X, self.Y)])
        assert self.approx(out, ref)

    def test_cov_btw(self):
        cov = BinaryCovarianceTW(
            self.n,
            estimator=hypernova.functional.pairedcorr
        )
        out = cov(self.X, self.Y)
        ref = np.stack([np.corrcoef(x, y)[:13, -7:]
                        for x, y in zip(self.X, self.Y)])
        assert self.approx(out, ref)

    def test_cov_bw(self):
        cov = BinaryCovariance(
            self.n,
            estimator=hypernova.functional.pairedcorr
        )
        out = cov(self.X, self.Y)
        ref = np.stack([np.corrcoef(x, y)[:13, -7:]
                        for x, y in zip(self.X, self.Y)])
        assert self.approx(out, ref)
