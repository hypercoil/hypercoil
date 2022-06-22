# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for kernels and distances.
"""
import pytest
import torch
import numpy as np
from hypercoil.functional import (
    linear_kernel,
    polynomial_kernel,
    gaussian_kernel,
    rbf_kernel,
    sigmoid_kernel
)
from sklearn.metrics.pairwise import (
    linear_kernel as lk_ref,
    polynomial_kernel as pk_ref,
    rbf_kernel as gk_ref,
    sigmoid_kernel as sk_ref
)


class TestKernel:

    def test_linear_kernel(self):
        n, p = 30, 100
        X = torch.randn(10, n, p)
        Y = torch.randn(10, n, p)
        ref = np.stack([lk_ref(x) for x in X])
        out = linear_kernel(X)
        assert np.allclose(out, ref, atol=1e-5)
        ref = np.stack([lk_ref(x, y) for x, y in zip(X, Y)])
        out = linear_kernel(X, Y)
        assert np.allclose(out, ref, atol=1e-5)

    def test_polynomial_kernel(self):
        n, p = 30, 100
        X = torch.randn(n, p)
        Y = torch.randn(n, p)
        ref = linear_kernel(X, Y)
        out = polynomial_kernel(X, Y, gamma=1, order=1)
        assert torch.allclose(out, ref, atol=1e-5)
        ref = pk_ref(X, Y)
        out = polynomial_kernel(X, Y, r=1)
        assert np.allclose(out, ref, atol=1e-5)
        ref = pk_ref(X, Y, gamma=-1, degree=7, coef0=-100)
        out = polynomial_kernel(X, Y, gamma=-1, order=7, r=-100)
        assert np.allclose(out, ref, atol=1e-5)

    def test_sigmoid_kernel(self):
        n, p = 30, 100
        X = torch.randn(n, p)
        Y = torch.randn(n, p)
        ref = sk_ref(X, Y)
        out = sigmoid_kernel(X, Y, r=1)
        assert np.allclose(out, ref, atol=1e-5)
        ref = sk_ref(X, Y, gamma=0.71, coef0=-2)
        out = sigmoid_kernel(X, Y, gamma=0.71, r=-2)
        assert np.allclose(out, ref, atol=1e-5)

    def test_gaussian_kernel(self):
        n, p = 30, 100
        X = torch.randn(n, p)
        Y = torch.randn(n, p)
        ref = gk_ref(X, Y)
        out = gaussian_kernel(X, Y)
        assert np.allclose(out, ref, atol=1e-5)
        ref = gk_ref(X, Y, gamma=-2)
        out = rbf_kernel(X, Y, gamma=-2)
        assert np.allclose(out, ref, atol=1e-5)
        ref = gk_ref(X, Y, gamma=0.25)
        out = gaussian_kernel(X, Y, sigma=2)
        assert np.allclose(out, ref, atol=1e-5)

    def test_parameterised_kernel(self):
        X = torch.tensor([
            [0., 1., 0., 1., 2.],
            [2., 1., 1., 1., 0.],
            [0., 0., 0., 0., 0.]
        ]).t()

        theta = torch.tensor([1., 1., 1.])
        ref = lk_ref(X)
        out = linear_kernel(X, theta=theta)
        assert np.allclose(out, ref, atol=1e-5)

        theta = torch.tensor([1., 1., 0.])
        ref = lk_ref(X)
        out = linear_kernel(X, theta=theta)
        assert np.allclose(out, ref, atol=1e-5)

        theta = torch.tensor([1., 0., 0.])
        ref = (X[:, 0].view(-1, 1) @ X[:, 0].view(1, -1))
        out = linear_kernel(X, theta=theta)
        assert np.allclose(out, ref, atol=1e-5)

        theta = torch.tensor([0., 1., 0.])
        ref = (X[:, 1].view(-1, 1) @ X[:, 1].view(1, -1))
        out = linear_kernel(X, theta=theta)
        assert np.allclose(out, ref, atol=1e-5)
