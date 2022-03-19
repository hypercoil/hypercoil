# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for IIR filter initialisation
"""
import pytest
import numpy as np
import torch
from hypercoil.init.freqfilter import (
    FreqFilterSpec,
    freqfilter_init_,
    clamp_init_
)


class TestIIRFilter:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        N = torch.Tensor([[1], [4]])
        Wn = torch.Tensor([[0.1, 0.3], [0.4, 0.6]])
        self.filter_specs = [
            FreqFilterSpec(Wn=[0.1, 0.3]),
            FreqFilterSpec(Wn=Wn, N=N),
            FreqFilterSpec(Wn=Wn, ftype='ideal'),
            FreqFilterSpec(Wn=[0.1, 0.2], N=[2, 2], btype='lowpass'),
            FreqFilterSpec(Wn=Wn, N=N, ftype='cheby1', rp=0.1),
            FreqFilterSpec(Wn=Wn, N=N, ftype='cheby2', rs=20),
            FreqFilterSpec(Wn=Wn, N=N, ftype='ellip', rs=20, rp=0.1),
            FreqFilterSpec(Wn=Wn, ftype='randn')
        ]
        self.Z = torch.complex(torch.Tensor(21, 15, 50),
                               torch.Tensor(21, 15, 50))
        self.clamped_specs = [
            FreqFilterSpec(Wn=[0.1, 0.3]),
            FreqFilterSpec(Wn=Wn, clamps=[{}, {0.1: 1}]),
            FreqFilterSpec(Wn=[0.1, 0.3], clamps=[{0.1: 0, 0.5:1}]),
            FreqFilterSpec(Wn=Wn, N=N, clamps=[{0.05: 1, 0.1: 0},
                                              {0.2: 0, 0.5: 1}])
        ]
        self.P = torch.Tensor(6, 30)
        self.V = torch.Tensor(7)
        self.P2 = torch.Tensor(1, 30)
        self.V2 = torch.Tensor(0)
        self.Z2 = torch.complex(torch.Tensor(21, 6, 50),
                                torch.Tensor(21, 6, 50))

        if torch.cuda.is_available():
            self.ZC = self.Z.clone().cuda()
            self.PC = self.P.clone().cuda()
            self.VC = self.V.clone().cuda()
            self.Z2C = self.Z2.clone().cuda()
            self.P2C = self.P2.clone().cuda()
            self.V2C = self.V2.clone().cuda()

    def test_freqfilter(self):
        freqfilter_init_(self.Z, self.filter_specs)
        assert sum(
            [spec.n_filters for spec in self.filter_specs]
            ) == self.Z.size(-2)

    def test_clamps(self):
        clamp_init_(self.P, self.V, self.clamped_specs)
        clamp_init_(self.P2, self.V2, [self.clamped_specs[0]])
        freqfilter_init_(self.Z2, self.clamped_specs)

    @pytest.mark.cuda
    def test_freqfilter_cuda(self):
        freqfilter_init_(self.ZC, self.filter_specs)
        assert sum(
            [spec.n_filters for spec in self.filter_specs]
            ) == self.ZC.size(-2)

    @pytest.mark.cuda
    def test_clamps_cuda(self):
        clamp_init_(self.PC, self.VC, self.clamped_specs)
        clamp_init_(self.P2C, self.V2C, [self.clamped_specs[0]])
        freqfilter_init_(self.Z2C, self.clamped_specs)
