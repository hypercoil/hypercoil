# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for IIR filter initialisation
"""
import pytest
import numpy as np
import torch
from hypercoil.init.iirfilter import (
    IIRFilterSpec,
    iirfilter_init_,
    clamp_init_
)


class TestIIRFilter:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        N = torch.Tensor([[1], [4]])
        Wn = torch.Tensor([[0.1, 0.3], [0.4, 0.6]])
        self.filter_specs = [
            IIRFilterSpec(Wn=[0.1, 0.3]),
            IIRFilterSpec(Wn=Wn, N=N),
            IIRFilterSpec(Wn=Wn, ftype='ideal'),
            IIRFilterSpec(Wn=[0.1, 0.2], N=[2, 2], btype='lowpass'),
            IIRFilterSpec(Wn=Wn, N=N, ftype='cheby1', rp=0.1),
            IIRFilterSpec(Wn=Wn, N=N, ftype='cheby2', rs=20),
            IIRFilterSpec(Wn=Wn, N=N, ftype='ellip', rs=20, rp=0.1),
            IIRFilterSpec(Wn=Wn, ftype='randn')
        ]
        self.Z = torch.complex(torch.Tensor(21, 15, 50),
                               torch.Tensor(21, 15, 50))
        self.clamped_specs = [
            IIRFilterSpec(Wn=[0.1, 0.3]),
            IIRFilterSpec(Wn=Wn, clamps=[{}, {0.1: 1}]),
            IIRFilterSpec(Wn=[0.1, 0.3], clamps=[{0.1: 0, 0.5:1}]),
            IIRFilterSpec(Wn=Wn, N=N, clamps=[{0.05: 1, 0.1: 0},
                                              {0.2: 0, 0.5: 1}])
        ]
        self.P = torch.Tensor(6, 30)
        self.V = torch.Tensor(7)
        self.P2 = torch.Tensor(1, 30)
        self.V2 = torch.Tensor(0)
        self.Z2 = torch.complex(torch.Tensor(21, 6, 50),
                                torch.Tensor(21, 6, 50))

    def test_iirfilter(self):
        iirfilter_init_(self.Z, self.filter_specs)
        assert sum(
            [spec.n_filters for spec in self.filter_specs]
            ) == self.Z.size(-2)

    def test_clamps(self):
        clamp_init_(self.P, self.V, self.clamped_specs)
        clamp_init_(self.P2, self.V2, [self.clamped_specs[0]])
        iirfilter_init_(self.Z2, self.clamped_specs)
