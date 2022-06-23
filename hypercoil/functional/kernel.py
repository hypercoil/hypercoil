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
    elif theta.is_sparse:
        pass
    elif theta.dim() == 1 or theta.shape[-1] != theta.shape[-2]:
        theta = _embed_params_in_diagonal(theta)
    else:
        theta = _embed_params_in_sparse(theta)
    X0 = sparse_mm(X0, theta)
    return sparse_mm(X0, X1.transpose(0, 1))


def linear_kernel(X0, X1=None, theta=None):
    r"""
    Compute the parameterised linear kernel between input tensors.

    For tensors :math:`X_0` and :math:`X_1` containing observations in column
    vectors, the parameterised linear kernel is

    :math:`K_{\theta}(X_0, X_1) = X_0^\intercal \theta X_1`

    where :math:`\theta` is tge kernel parameter.

    :Dimension: **X0 :** :math:`(*, N, P)` or :math:`(N, P, *)`
                    N denotes number of observations, P denotes number of
                    features, `*` denotes any number of additional dimensions.
                    If the input is dense, then the last dimensions should be
                    N and P; if it is sparse, then the first dimensions should
                    be N and P.
                **X1 :** :math:`(*, M, P)` or  :math:`(M, P, *)`
                    M denotes number of observations.
                **theta :** :math:`(*, P, P)` or :math:`(*, P)`
                    As above.
                **Output :** :math:`(*, M, N)` or :math:`(M, N, *)`
                    As above.

    Parameters
    ----------
    X0 : tensor
        A feature tensor.
    X1 : tensor or None
        Second feature tensor. If not explicitly provided, the kernel of
        ``X`` with itself is computed.
    theta : tensor or None
        Kernel parameter (generally a representation of a positive definite
        matrix). If not provided, defaults to identity (an unparameterised
        kernel). If the last two dimensions are the same size, they are used
        as a matrix parameter; if they are not, the final axis is instead
        used as the diagonal of the matrix.

    Returns
    -------
    tensor
        Kernel Gram matrix.
    """
    if X1 is None:
        X1 = X0
    if X0.is_sparse:
        return _linear_kernel_sparse(X0, X1, theta)
    else:
        return _linear_kernel_dense(X0, X1, theta)


def linear_distance(X0, X1=None, theta=None):
    """Squared Euclidean (L2) distance."""
    if X1 is None:
        X1 = X0
    D = X0.unsqueeze(-2) - X1.unsqueeze(-3)
    D = linear_kernel(X0=D.unsqueeze(-2), theta=theta)
    return D.view(*D.shape[:-2])


def polynomial_kernel(X0, X1=None, theta=None, gamma=None, order=3, r=0):
    r"""
    Compute the parameterised polynomial kernel between input tensors.

    For tensors :math:`X_0` and :math:`X_1` containing observations in column
    vectors, the parameterised polynomial kernel is

    :math:`K_{\theta}(X_0, X_1) = (\gamma X_0^\intercal \theta X_1 + r)^\omega`

    where :math:`\theta` is the kernel parameter, :math:`\gamma` and r are
    scaling and offset coefficients, and :math:`\omega` is the maximum degree
    or order of the kernel.

    :Dimension: **X0 :** :math:`(*, N, P)` or :math:`(N, P, *)`
                    N denotes number of observations, P denotes number of
                    features, `*` denotes any number of additional dimensions.
                    If the input is dense, then the last dimensions should be
                    N and P; if it is sparse, then the first dimensions should
                    be N and P.
                **X1 :** :math:`(*, M, P)` or  :math:`(M, P, *)`
                    M denotes number of observations.
                **theta :** :math:`(*, P, P)` or :math:`(*, P)`
                    As above.
                **Output :** :math:`(*, M, N)` or :math:`(M, N, *)`
                    As above.

    Parameters
    ----------
    X0 : tensor
        A feature tensor.
    X1 : tensor or None
        Second feature tensor. If not explicitly provided, the kernel of
        ``X`` with itself is computed.
    theta : tensor or None
        Kernel parameter (generally a representation of a positive definite
        matrix). If not provided, defaults to identity (an unparameterised
        kernel). If the last two dimensions are the same size, they are used
        as a matrix parameter; if they are not, the final axis is instead
        used as the diagonal of the matrix.
    gamma : float or None (default None)
        Scaling coefficient. If not explicitly specified, this is
        automatically set to :math:`\frac{1}{P}`.
    order : int (default 3)
        Maximum degree of the polynomial.
    r : float (default 0)
        Offset coefficient.

    Returns
    -------
    tensor
        Kernel Gram matrix.
    """
    gamma = _default_gamma(X0, gamma)
    K = linear_kernel(X0, X1, theta)
    return (gamma * K + r) ** order


def sigmoid_kernel(X0, X1=None, theta=None, gamma=None, r=0):
    r"""
    Compute the parameterised sigmoid kernel between input tensors.

    For tensors :math:`X_0` and :math:`X_1` containing observations in column
    vectors, the parameterised sigmoid kernel is

    :math:`K_{\theta}(X_0, X_1) = \tanh (\gamma X_0^\intercal \theta X_1 + r)`

    where :math:`\theta` is the kernel parameter, and :math:`\gamma` and r are
    scaling and offset coefficients.

    :Dimension: **X0 :** :math:`(*, N, P)` or :math:`(N, P, *)`
                    N denotes number of observations, P denotes number of
                    features, `*` denotes any number of additional dimensions.
                    If the input is dense, then the last dimensions should be
                    N and P; if it is sparse, then the first dimensions should
                    be N and P.
                **X1 :** :math:`(*, M, P)` or  :math:`(M, P, *)`
                    M denotes number of observations.
                **theta :** :math:`(*, P, P)` or :math:`(*, P)`
                    As above.
                **Output :** :math:`(*, M, N)` or :math:`(M, N, *)`
                    As above.

    Parameters
    ----------
    X0 : tensor
        A feature tensor.
    X1 : tensor or None
        Second feature tensor. If not explicitly provided, the kernel of
        ``X`` with itself is computed.
    theta : tensor or None
        Kernel parameter (generally a representation of a positive definite
        matrix). If not provided, defaults to identity (an unparameterised
        kernel). If the last two dimensions are the same size, they are used
        as a matrix parameter; if they are not, the final axis is instead
        used as the diagonal of the matrix.
    gamma : float or None (default None)
        Scaling coefficient. If not explicitly specified, this is
        automatically set to :math:`\frac{1}{P}`.
    r : float (default 0)
        Offset coefficient.

    Returns
    -------
    tensor
        Kernel Gram matrix.
    """
    gamma = _default_gamma(X0, gamma)
    K = linear_kernel(X0, X1, theta)
    return torch.tanh(gamma * K + r)


def gaussian_kernel(X0, X1=None, theta=None, sigma=None):
    r"""
    Compute the parameterised Gaussian kernel between input tensors.

    For tensors :math:`X_0` and :math:`X_1` containing observations in column
    vectors, the parameterised Gaussian kernel is

    :math:`K_{\theta}(X_0, X_1) = e^{\frac{1}{\sigma^2} (X_0 - X_1)^\intercal \theta (X_0 - X_1)}`

    where :math:`\theta` is the kernel parameter, :math:`\sigma` is an
    isotropic standard deviation, and :math:`X_0 - X_1` contains all pairwise
    differences between vectors in :math:`X_0` and :math:`X_1`. The kernel
    parameter :math:`\theta` can also be interpreted as an inverse covariance.

    This is the same as :func:`rbf_kernel` but is parameterised in terms of
    :math:`\sigma` rather than  :math:`\gamma`.

    :Dimension: **X0 :** :math:`(*, N, P)` or :math:`(N, P, *)`
                    N denotes number of observations, P denotes number of
                    features, `*` denotes any number of additional dimensions.
                    If the input is dense, then the last dimensions should be
                    N and P; if it is sparse, then the first dimensions should
                    be N and P.
                **X1 :** :math:`(*, M, P)` or  :math:`(M, P, *)`
                    M denotes number of observations.
                **theta :** :math:`(*, P, P)` or :math:`(*, P)`
                    As above.
                **Output :** :math:`(*, M, N)` or :math:`(M, N, *)`
                    As above.

    Parameters
    ----------
    X0 : tensor
        A feature tensor.
    X1 : tensor or None
        Second feature tensor. If not explicitly provided, the kernel of
        ``X`` with itself is computed.
    theta : tensor or None
        Kernel parameter (generally a representation of a positive definite
        matrix). If not provided, defaults to identity (an unparameterised
        kernel). If the last two dimensions are the same size, they are used
        as a matrix parameter; if they are not, the final axis is instead
        used as the diagonal of the matrix.
    gamma : float or None (default None)
        Scaling coefficient. If not explicitly specified, this is
        automatically set to :math:`\frac{1}{P}`.

    Returns
    -------
    tensor
        Kernel Gram matrix.
    """
    if sigma is not None:
        gamma = sigma ** -2
    else:
        gamma = sigma
    return rbf_kernel(X0, X1, theta=theta, gamma=gamma)


def rbf_kernel(X0, X1=None, theta=None, gamma=None):
    r"""
    Compute the parameterised RBF kernel between input tensors.

    For tensors :math:`X_0` and :math:`X_1` containing observations in column
    vectors, the parameterised RBF kernel is

    :math:`K_{\theta}(X_0, X_1) = e^{\gamma (X_0 - X_1)^\intercal \theta (X_0 - X_1)}`

    where :math:`\theta` is the kernel parameter, :math:`\gamma` is a scaling
    coefficient, and :math:`X_0 - X_1` contains all pairwise differences
    between vectors in :math:`X_0` and :math:`X_1`. The kernel parameter
    :math:`\theta` can also be interpreted as an inverse covariance.

    This is the same as :func:`gaussian_kernel` but is parameterised in terms
    of :math:`\gamma` rather than  :math:`\sigma`.

    :Dimension: **X0 :** :math:`(*, N, P)` or :math:`(N, P, *)`
                    N denotes number of observations, P denotes number of
                    features, `*` denotes any number of additional dimensions.
                    If the input is dense, then the last dimensions should be
                    N and P; if it is sparse, then the first dimensions should
                    be N and P.
                **X1 :** :math:`(*, M, P)` or  :math:`(M, P, *)`
                    M denotes number of observations.
                **theta :** :math:`(*, P, P)` or :math:`(*, P)`
                    As above.
                **Output :** :math:`(*, M, N)` or :math:`(M, N, *)`
                    As above.

    Parameters
    ----------
    X0 : tensor
        A feature tensor.
    X1 : tensor or None
        Second feature tensor. If not explicitly provided, the kernel of
        ``X`` with itself is computed.
    theta : tensor or None
        Kernel parameter (generally a representation of a positive definite
        matrix). If not provided, defaults to identity (an unparameterised
        kernel). If the last two dimensions are the same size, they are used
        as a matrix parameter; if they are not, the final axis is instead
        used as the diagonal of the matrix.
    gamma : float or None (default None)
        Scaling coefficient. If not explicitly specified, this is
        automatically set to :math:`\frac{1}{P}`.

    Returns
    -------
    tensor
        Kernel Gram matrix.
    """
    gamma = _default_gamma(X0, gamma)
    K = linear_distance(X0, X1, theta)
    return torch.exp(-gamma * K)
