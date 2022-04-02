# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for noise sources
"""
import pytest
import numpy as np
import torch
from hypercoil.functional import (
    SPSDNoiseSource,
    LowRankNoiseSource,
    BandDropoutSource,
    UnstructuredNoiseSource
)


#TODO: There are many missing tests for noise and dropout sources.
# We should have at minimum an injection test for each source
# on CPU and CUDA that also verifies each source works given an input
# of a reasonable but nontrivial shape.


def lr_std_mean(dim=100, rank=None, var=0.05, iter=1000):
    lrns = LowRankNoiseSource(rank=rank, var=var)
    return torch.Tensor(
        [lrns.sample([dim]).std() for _ in range(iter)
    ]).mean()


class TestNoise:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.atol = 1e-3
        self.rtol = 1e-4
        self.approx = lambda out, ref: np.isclose(
            out, ref, atol=self.atol, rtol=self.rtol)

    def test_lr_std(self):
        out = lr_std_mean()
        ref = 0.05
        assert self.approx(out, ref)
        out = lr_std_mean(var=0.2)
        ref = 0.2
        assert self.approx(out, ref)
        out = lr_std_mean(var=0.03, rank=7)
        ref = 0.03
        assert self.approx(out, ref)

    def test_spsd_spsd(self):
        spsdns = SPSDNoiseSource()
        out = spsdns.sample([100])
        assert torch.allclose(out, out.T, atol=1e-5)
        L = torch.linalg.eigvalsh(out)
        # ignore effectively-zero eigenvalues
        L[torch.abs(L) < 1e-4] = 0
        assert L.min() >= 0
        assert torch.all(L >= 0)

    def test_band_correction(self):
        bds = BandDropoutSource()
        out = bds.sample([100]).sum()
        ref = bds.bandmask.sum()
        assert torch.abs((out - ref) / ref) <= 0.2

    def test_scalar_iid_noise(self):
        sz = torch.Size([3, 8, 1, 21, 1])
        inp = torch.rand(sz)
        sins = UnstructuredNoiseSource()
        out = sins(inp)
        assert out.size() == sz

    @pytest.mark.cuda
    def test_scalar_iid_noise_cuda(self):
        sz = torch.Size([3, 8, 1, 21, 1])
        inp = torch.rand(sz, device='cuda')
        sins = UnstructuredNoiseSource()
        out = sins(inp)
        assert out.size() == sz
