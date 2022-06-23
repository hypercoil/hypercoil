# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Parameterised similarity kernels and distance metrics.
"""
import torch
from .utils import sparse_mm, _conform_vector_weight, _promote_nnz_dim


def _default_gamma(X, gamma):
    if gamma is None:
        gamma = 1 / X.size(-1)
    return gamma


def _embed_params_in_diagonal(theta):
    dim = theta.size(-1)
    indices = torch.arange(dim, dtype=theta.dtype, device=theta.device)
    indices = torch.stack((indices, indices))
    theta = _promote_nnz_dim(theta)
    return torch.sparse_coo_tensor(
        indices, theta, size=(dim, dim, *theta.shape[1:])
    )


def _embed_params_in_sparse(theta):
    dim = theta.size(-1)
    with torch.no_grad():
        nzi = theta.abs().sum(list(range(theta.dim() - 2)))
        indices = torch.stack(torch.where(nzi))
    values = _promote_nnz_dim(theta[..., indices[0], indices[1]])
    return torch.sparse_coo_tensor(
        indices, values, size=(dim, dim, *values.shape[1:])
    )


def _linear_kernel_dense(X0, X1, theta):
    if theta is None:
        return X0 @ X1.transpose(-1, -2)
    elif theta.dim() == 1 or theta.shape[-1] != theta.shape[-2]:
        theta = _conform_vector_weight(theta)
        return (X0 * theta) @ X1.transpose(-1, -2)
    else:
        return X0 @ theta @ X1.transpose(-1, -2)


def _linear_kernel_sparse(X0, X1, theta):
    if theta is None:
        return sparse_mm(X0, X1.transpose(0, 1))
    elif theta.dim() == 1 or theta.shape[-1] != theta.shape[-2]:
        theta = _embed_params_in_diagonal(theta)
        #shape = [1 for _ in range(1, values.dim())]
        #theta = theta.view(-1, *shape)
        #print(values.shape, theta.shape)
        #values = values * theta
    else:
        theta = _embed_params_in_sparse(theta)
    #indices = X0._indices()
    #X0 = torch.sparse_coo_tensor(
    #    indices=indices, values=values, size=(X0.size()))
    X0 = sparse_mm(X0, theta)
    return sparse_mm(X0, X1.transpose(0, 1))


def linear_kernel(X0, X1=None, theta=None):
    if X1 is None:
        X1 = X0
    if X0.is_sparse:
        return _linear_kernel_sparse(X0, X1, theta)
    else:
        return _linear_kernel_dense(X0, X1, theta)


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
