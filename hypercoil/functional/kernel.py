# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Parameterised similarity kernels and distance metrics.
"""
import torch


def _default_gamma(X, gamma):
    if gamma is None:
        gamma = 1 / X.size(-1)
    return gamma


def linear_kernel(X0, X1=None, theta=None):
    if X1 is None:
        X1 = X0
    if theta is None:
        return X0 @ X1.transpose(-1, -2)
    else:
        return X0 @ theta @ X1.transpose(-1, -2)


def linear_distance(X0, X1=None, theta=None):
    if X1 is None:
        X1 = X0
    D = X0.unsqueeze(-2) - X1.unsqueeze(-3)
    D = linear_kernel(X0=D.unsqueeze(-2), theta=theta)
    return D.view(*D.shape[:-2])


def polynomial_kernel(X0, X1=None, theta=None, gamma=None, order=3, r=0):
    gamma = _default_gamma(X0, gamma)
    K = linear_kernel(X0, X1, theta)
    return (gamma * K + r) ** order


def sigmoid_kernel(X0, X1=None, theta=None, gamma=None, r=0):
    gamma = _default_gamma(X0, gamma)
    K = linear_kernel(X0, X1, theta)
    return torch.tanh(gamma * K + r)


def gaussian_kernel(X0, X1=None, theta=None, sigma=None):
    if sigma is not None:
        gamma = sigma ** -2
    else:
        gamma = sigma
    return rbf_kernel(X0, X1, theta=theta, gamma=gamma)


def rbf_kernel(X0, X1=None, theta=None, gamma=None):
    gamma = _default_gamma(X0, gamma)
    K = linear_distance(X0, X1, theta)
    return torch.exp(-gamma * K)
