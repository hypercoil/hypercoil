# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for noise sources
"""
import numpy as np
import torch
from hypernova.functional import (
    SPSDNoiseSource
)


atol = 1e-3
rtol = 1e-4
testf = lambda out, ref: np.isclose(out, ref, atol=atol, rtol=rtol)


def spsd_std_mean(dim=100, rank=None, std=0.05, iter=1000):
    spsdns = SPSDNoiseSource(rank=rank, std=std)
    return torch.Tensor(
        [spsdns.sample([dim]).std() for _ in range(iter)
    ]).mean()


def spsd_mean_mean(dim=100, rank=None, std=0.05, iter=1000):
    spsdns = SPSDNoiseSource(rank=rank, std=std)
    return torch.Tensor(
        [spsdns.sample([dim]).mean() for _ in range(iter)
    ]).mean()


def test_spsd_std():
    out = spsd_std_mean()
    ref = 0.05
    assert testf(out, ref)
    out = spsd_std_mean(std=0.2)
    ref = 0.2
    assert testf(out, ref)
    out = spsd_std_mean(std=0.03, rank=7)
    ref = 0.03
    assert testf(out, ref)


def test_spsd_spsd():
    spsdns = SPSDNoiseSource()
    out = spsdns.sample([100])
    assert np.allclose(out, out.T, atol=1e-7)
    # ignore effectively-zero eigenvalues
    L = np.linalg.eigvals(out)
    L[np.abs(L) < 1e-7] = 0
    assert np.all(L >= 0)

    spsdns = SPSDNoiseSource(rank=3)
    out = spsdns.sample([100])
    assert np.allclose(out, out.T, atol=1e-7)
    # ignore effectively-zero eigenvalues
    L = np.linalg.eigvals(out)
    L[np.abs(L) < 1e-7] = 0
    assert np.all(L >= 0)
