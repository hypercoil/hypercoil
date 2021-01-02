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
from hypernova.functional.noise import (
    DiagonalNoiseSource, DiagonalDropoutSource, BandDropoutSource
)
from hypernova.functional.domain import Logit


class TestCovNN:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.approx = lambda x, y: np.allclose(x.detach(), y, atol=1e-5)

        self.X = torch.rand(4, 13, 100)
        self.Y = torch.rand(4, 7, 100)
        self.dns = DiagonalNoiseSource()
        self.dds = DiagonalDropoutSource()
        self.bds = BandDropoutSource()

    def test_cov_uuw(self):
        cov = UnaryCovarianceUW(100, estimator=hypernova.functional.corr)
        out = cov(self.X)
        ref = np.stack([np.corrcoef(x) for x in self.X])
        assert self.approx(out, ref)
        cov = UnaryCovarianceUW(
            100, estimator=hypernova.functional.pcorr,
            out_channels=7, noise=self.dns) #, dropout=dds)
        cov(self.X)

    def test_cov_utw(self):
        cov = UnaryCovarianceTW(100, estimator=hypernova.functional.corr)
        out = cov(self.X)
        ref = np.stack([np.corrcoef(x) for x in self.X])
        assert self.approx(out, ref)
        cov = UnaryCovarianceTW(
            100, estimator=hypernova.functional.pcorr, max_lag=3,
            out_channels=7, noise=self.dns, dropout=self.bds, domain=Logit(4))
        cov(self.X)
        assert cov.prepreweight_c.size() == torch.Size([4, 7])
        assert cov.weight[5, 15, 17] == cov.weight[5, 94, 96]
        assert cov.postweight[3, 13, 17] == 0

    def test_cov_uw(self):
        cov = UnaryCovariance(100, estimator=hypernova.functional.corr)
        out = cov(self.X)
        ref = np.stack([np.corrcoef(x) for x in self.X])
        assert self.approx(out, ref)

    def test_cov_buw(self):
        cov = BinaryCovarianceUW(
            100, estimator=hypernova.functional.pairedcorr)
        out = cov(self.X, self.Y)
        ref = np.stack([np.corrcoef(x, y)[:13, -7:]
                        for x, y in zip(self.X, self.Y)])
        assert self.approx(out, ref)

    def test_cov_btw(self):
        cov = BinaryCovarianceTW(100, estimator=hypernova.functional.pairedcorr)
        out = cov(self.X, self.Y)
        ref = np.stack([np.corrcoef(x, y)[:13, -7:]
                        for x, y in zip(self.X, self.Y)])
        assert self.approx(out, ref)

    def test_cov_bw(self):
        cov = BinaryCovariance(100, estimator=hypernova.functional.pairedcorr)
        out = cov(self.X, self.Y)
        ref = np.stack([np.corrcoef(x, y)[:13, -7:]
                        for x, y in zip(self.X, self.Y)])
        assert self.approx(out, ref)
