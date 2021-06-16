# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for polynomial convolution layer
"""
import pytest
import numpy as np
import torch
from hypercoil.nn import PolyConv2D
from hypercoil.init.deltaplus import DeltaPlusInit


class TestPolyConv:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.X = torch.rand(4, 13, 100)

        self.approx = torch.allclose

    def test_polyconv_identity(self):
        init = DeltaPlusInit(var=0, loc=(0, 0, 3))
        poly = PolyConv2D(2, 4, init=init)
        out = poly(self.X)
        ref = self.X.unsqueeze(1).repeat(1, 4, 1, 1)
        assert self.approx(out, ref)

    def test_polyconv_shapes(self):
        poly = PolyConv2D(7, 3)
        out = poly(self.X).size()
        ref = torch.Size([4, 3, 13, 100])
        assert out == ref
