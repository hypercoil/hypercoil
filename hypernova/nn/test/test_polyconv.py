# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for polynomial convolution layer
"""
import numpy as np
import torch
from hypernova.nn import PolyConv2D
from hypernova.init.deltaplus import deltaplus_init_


testf = torch.allclose


X = torch.rand(4, 13, 100)


def test_polyconv_identity():
    poly = PolyConv2D(2, 4, init_=deltaplus_init_, init_params={'var': 0})
    out = poly(X)
    ref = X.unsqueeze(1).repeat(1, 4, 1, 1)
    assert testf(out, ref)


def test_polyconv_shapes():
    poly = PolyConv2D(7, 3)
    out = poly(X).size()
    ref = torch.Size([4, 3, 13, 100])
    assert out == ref
