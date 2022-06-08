# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for activation functions
"""
import pytest
import torch
from hypercoil.functional import cov, corr
from hypercoil.functional.activation import corrnorm


class TestActivationFunctions:

    def test_corrnorm(self):
        A = torch.rand(12, 30)
        assert torch.all(corrnorm(cov(A)) == corr(A))
        B = torch.randn(12, 12)
        B = B + B.t()
        assert torch.all(
            torch.sign(torch.diagonal(B)) ==
            torch.sign(torch.diagonal(corrnorm(B)))
        )
