# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for covariance modules
"""
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


testf = lambda x, y: np.allclose(x.detach(), y, atol=1e-5)


X = torch.rand(4, 13, 100)
Y = torch.rand(4, 7, 100)
dns = DiagonalNoiseSource()
dds = DiagonalDropoutSource()
bds = BandDropoutSource()


def test_cov_uuw():
    cov = UnaryCovarianceUW(100, estimator=hypernova.functional.corr)
    out = cov(X)
    ref = np.stack([np.corrcoef(x) for x in X])
    assert testf(out, ref)
    cov = UnaryCovarianceUW(
        100, estimator=hypernova.functional.pcorr,
        out_channels=7, noise=dns) #, dropout=dds)
    cov(X)


def test_cov_utw():
    cov = UnaryCovarianceTW(100, estimator=hypernova.functional.corr)
    out = cov(X)
    ref = np.stack([np.corrcoef(x) for x in X])
    assert testf(out, ref)
    cov = UnaryCovarianceTW(
        100, estimator=hypernova.functional.pcorr, max_lag=3,
        out_channels=7, noise=dns, dropout=bds, domain=Logit(4))
    cov(X)
    assert cov.prepreweight_c.size() == torch.Size([4, 7])
    assert cov.weight[5, 15, 17] == cov.weight[5, 94, 96]
    assert cov.postweight[3, 13, 17] == 0


def test_cov_uw():
    cov = UnaryCovariance(100, estimator=hypernova.functional.corr)
    out = cov(X)
    ref = np.stack([np.corrcoef(x) for x in X])
    assert testf(out, ref)


def test_cov_buw():
    cov = BinaryCovarianceUW(100, estimator=hypernova.functional.pairedcorr)
    out = cov(X, Y)
    ref = np.stack([np.corrcoef(x, y)[:13, -7:] for x, y in zip(X, Y)])
    assert testf(out, ref)


def test_cov_btw():
    cov = BinaryCovarianceTW(100, estimator=hypernova.functional.pairedcorr)
    out = cov(X, Y)
    ref = np.stack([np.corrcoef(x, y)[:13, -7:] for x, y in zip(X, Y)])
    assert testf(out, ref)


def test_cov_bw():
    cov = BinaryCovariance(100, estimator=hypernova.functional.pairedcorr)
    out = cov(X, Y)
    ref = np.stack([np.corrcoef(x, y)[:13, -7:] for x, y in zip(X, Y)])
    assert testf(out, ref)
