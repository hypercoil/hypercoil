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
from hypernova import (
    cov, corr, partialcorr, pairedcov, precision,
    conditionalcov, conditionalcorr
)


tol = 5e-7
ofunc = cov
rfunc = np.cov
testf = lambda out, ref: np.allclose(out, ref, atol=tol)

X = np.random.rand(7, 100)
Xt = torch.Tensor(X)


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
