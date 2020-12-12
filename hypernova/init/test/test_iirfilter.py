# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for IIR filter initialisation
"""
import numpy as np
import torch
from hypernova.nn import PolyConv2D
from hypernova.init.iirfilter import (
    IIRFilterSpec,
    iirfilter_init_
)


testf = torch.allclose


N = torch.Tensor([[1], [4]])
Wn = torch.Tensor([[0.1, 0.3], [0.4, 0.6]])
filter_specs = [
    IIRFilterSpec(Wn=[0.1, 0.3]),
    IIRFilterSpec(Wn=Wn, N=N),
    IIRFilterSpec(Wn=Wn, ftype='ideal'),
    IIRFilterSpec(Wn=[0.1, 0.2], N=[2, 2], btype='lowpass'),
    IIRFilterSpec(Wn=Wn, N=N, ftype='cheby1', rp=0.1),
    IIRFilterSpec(Wn=Wn, N=N, ftype='cheby2', rs=20),
    IIRFilterSpec(Wn=Wn, N=N, ftype='cheby2', rs=20, rp=0.1)
]
Z = torch.complex(torch.Tensor(21, 13, 50), torch.Tensor(21, 13, 50))


def test_iirfilter():
    iirfilter_init_(Z, filter_specs)
