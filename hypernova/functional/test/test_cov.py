# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for covariance, correlation, and derived measures
"""
import numpy as np
import pandas as pd
import torch
import pingouin
from hypernova.functional import (
    cov, corr, partialcorr, pairedcov, precision,
    conditionalcov, conditionalcorr
)


tol = 5e-7
ofunc = cov
rfunc = np.cov
testf = lambda out, ref: np.allclose(out, ref, atol=tol)

x = np.random.rand(100)
X = np.random.rand(7, 100)
XM = np.random.rand(200, 7, 100)
w = np.random.rand(100)
W = np.random.rand(100, 100)
Y = np.random.rand(3, 100)
xt = torch.Tensor(x)
Xt = torch.Tensor(X)
XMt = torch.Tensor(XM)
wt = torch.Tensor(w)
Wt = torch.Tensor(W)
Yt = torch.Tensor(Y)


def covpattern(**args):
    out = ofunc(Xt, **args).numpy()
    ref = rfunc(X, **args)
    assert testf(out, ref)


def test_cov_vanilla():
    args = {}
    covpattern(**args)


def test_cov_transpose():
    args = {'rowvar': False}
    covpattern(**args)


def test_cov_biased():
    args = {'bias': True}
    covpattern(**args)


def test_cov_custdof():
    args = {'ddof': 17}
    covpattern(**args)


def test_cov_var():
    out = ofunc(xt).numpy()
    ref = rfunc(x)
    assert testf(out, ref)


def test_cov_weighted():
    out = ofunc(Xt, weight=wt).numpy()
    ref = rfunc(X, aweights=w)
    assert testf(out, ref)


def test_cov_Weighted():
    out = ofunc(Xt, weight=Wt)


def test_cov_multidim():
    out = ofunc(XMt, weight=wt).numpy()
    ref = np.stack([
        rfunc(XM[i, :, :].squeeze(), aweights=w)
        for i in range(XM.shape[0])
    ])
    assert testf(out, ref)


def test_paired():
    out = pairedcov(Xt, Yt).numpy()
    ref = np.cov(np.concatenate([X ,Y], -2))[:7, -3:]
    assert testf(out, ref)


def test_corr():
    out = corr(Xt).numpy()
    ref = np.corrcoef(X)
    assert testf(out, ref)


def test_pcorr():
    out = partialcorr(Xt).numpy()
    ref = pd.DataFrame(X.T).pcorr().values
    assert testf(out, ref)


def test_ccov():
    """
    Verify equivalence of the Schur complement approach and fit-based confound
    regression.
    """
    out = conditionalcov(Xt, Yt).numpy()
    ref = torch.pinverse(precision(torch.cat([Xt ,Yt], -2))[:7, :7]).numpy()
    assert testf(out, ref)
    Y_intercept = np.concatenate([Y, np.ones((1, 100))])
    ref = np.cov(
        X - np.linalg.lstsq(Y_intercept.T, X.T, rcond=None)[0].T
        @ Y_intercept)
    assert testf(out, ref)


def test_ccorr():
    """
    Verify equivalence of the Schur complement approach and fit-based confound
    regression.
    """
    Y_intercept = np.concatenate([Y, np.ones((1, 100))])
    out = conditionalcorr(Xt, Yt).numpy()
    ref = np.corrcoef(
        X - np.linalg.lstsq(Y_intercept.T, X.T, rcond=None)[0].T
        @ Y_intercept)
    assert testf(out, ref)
