# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for covariance, correlation, and derived measures
"""
import pytest
import numpy as np
import pandas as pd
import torch
import pingouin
from hypernova.functional import (
    cov, corr, partialcorr, pairedcov, pairedcorr, precision,
    conditionalcov, conditionalcorr
)


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
        self.W = np.random.rand(100, 100)
        self.Y = np.random.rand(3, 100)
        self.xt = torch.Tensor(self.x)
        self.Xt = torch.Tensor(self.X)
        self.XMt = torch.Tensor(self.XM)
        self.wt = torch.Tensor(self.w)
        self.wMt = torch.Tensor(self.wM)
        self.Wt = torch.Tensor(self.W)
        self.WMt = torch.diag_embed(self.wMt)
        self.Yt = torch.Tensor(self.Y)

    def covpattern(self, **args):
        out = self.ofunc(self.Xt, **args).numpy()
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
        out = self.ofunc(self.xt).numpy()
        ref = self.rfunc(self.x)
        assert self.approx(out, ref)

    def test_cov_weighted(self):
        out = self.ofunc(self.Xt, weight=self.wt).numpy()
        ref = self.rfunc(self.X, aweights=self.w)
        assert self.approx(out, ref)

    def test_cov_multiweighted_1d(self):
        out = self.ofunc(self.Xt, weight=self.wMt)
        ref = np.stack([
            self.rfunc(self.X, aweights=self.wM[i, :])
            for i in range(self.wM.shape[0])
        ])
        assert self.approx(out, ref)

    def test_cov_multiweighted(self):
        out = cov(self.Xt, weight=self.WMt)
        ref = cov(self.Xt, weight=self.WMt[0, :])
        assert self.approx(out[0, :], ref)
        assert out.size() == torch.Size([3, 7, 7])

    def test_cov_Weighted(self):
        out = self.ofunc(self.Xt, weight=self.Wt)

    def test_cov_multidim(self):
        out = self.ofunc(self.XMt, weight=self.wt).numpy()
        ref = np.stack([
            self.rfunc(self.XM[i, :, :].squeeze(), aweights=self.w)
            for i in range(self.XM.shape[0])
        ])
        assert self.approx(out, ref)

    def test_paired(self):
        out = pairedcov(self.Xt, self.Yt).numpy()
        ref = np.cov(np.concatenate([self.X ,self.Y], -2))[:7, -3:]
        assert self.approx(out, ref)

    def test_corr(self):
        out = corr(self.Xt).numpy()
        ref = np.corrcoef(self.X)
        assert self.approx(out, ref)

    def test_pairedcorr(self):
        out = pairedcorr(self.Xt, self.Yt).numpy()
        ref = corr(torch.cat([self.Xt, self.Yt]))[:7, 7:].numpy()
        assert self.approx(out, ref)

    def test_pcorr(self):
        out = partialcorr(self.Xt).numpy()
        ref = pd.DataFrame(self.X.T).pcorr().values
        assert self.approx(out, ref)

    def test_ccov(self):
        """
        Verify equivalence of the Schur complement approach and fit-based
        confound regression.
        """
        out = conditionalcov(self.Xt, self.Yt).numpy()
        ref = torch.pinverse(
            precision(torch.cat([self.Xt ,self.Yt], -2))[:7, :7]).numpy()
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
        out = conditionalcorr(self.Xt, self.Yt).numpy()
        ref = np.corrcoef(
            self.X - np.linalg.lstsq(Y_intercept.T, self.X.T, rcond=None)[0].T
            @ Y_intercept)
        assert self.approx(out, ref)
