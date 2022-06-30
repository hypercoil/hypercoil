# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for activation functions
"""
import pytest
import torch
from hypercoil.functional import cov, corr
from hypercoil.functional.activation import corrnorm, isochor


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

    def test_isochor(self):
        # on rare occasions it's outside tolerance
        torch.random.manual_seed(238)
        A = torch.randn(5, 20, 20)
        A = A @ A.transpose(-1, -2)
        out = isochor(A)
        assert torch.allclose(
            torch.linalg.det(out), torch.tensor(1.), rtol=1e-2)

        out = isochor(A, volume=4)
        assert torch.allclose(
            torch.linalg.det(out), torch.tensor(4.), rtol=1e-2)

        out = isochor(A, volume=4, max_condition=5)
        assert torch.allclose(
            torch.linalg.det(out), torch.tensor(4.), rtol=1e-2)
        L, Q = torch.linalg.eigh(out)
        print(L.amax(dim=-1), L.amin(dim=-1))
        assert torch.all((L.amax(dim=-1) / L.amin(dim=-1)) < 5.01)

        out = isochor(A, softmax_temp=1e10)
        assert torch.allclose(out, torch.eye(20), atol=1e-6)

        out = isochor(A, volume=(2 ** 20), max_condition=1)
        assert torch.allclose(out, 2 * torch.eye(20), atol=1e-6)
