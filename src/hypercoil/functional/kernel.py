# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Parameterised similarity kernels and distance metrics.
"""
from __future__ import annotations
from functools import singledispatch
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO

from ..engine import NestedDocParse, Tensor
from .cov import pairedcorr, pairedcov
from .sparse import (
    TopKTensor,
    spdiagmm,
    spsp_innerpaired,
    spsp_pairdiff,
    spspmm,
    topkx,
)
from .utils import _conform_vector_weight, is_sparse


def _default_gamma(X: Tensor, *, gamma: Optional[float]) -> float:
    if gamma is None:
        gamma = 1 / X.shape[-1]
    return gamma


def document_param_pairwise(f: Callable) -> Callable:
    linear_kernel_long_desc = r"""
    For tensors :math:`X_0` and :math:`X_1` containing features in column
    vectors, the parameterised linear kernel is

    :math:`K_{\theta}(X_0, X_1) = X_0^\intercal \theta X_1`

    where :math:`\theta` is the kernel parameter."""

    polynomial_kernel_long_desc = r"""
    For tensors :math:`X_0` and :math:`X_1` containing features in column
    vectors, the parameterised polynomial kernel is

    :math:`K_{\theta}(X_0, X_1) = (\gamma X_0^\intercal \theta X_1 + r)^\omega`

    where :math:`\theta` is the kernel parameter, :math:`\gamma` and r are
    scaling and offset coefficients, and :math:`\omega` is the maximum degree
    or order of the kernel."""

    sigmoid_kernel_long_desc = r"""
    For tensors :math:`X_0` and :math:`X_1` containing features in column
    vectors, the parameterised sigmoid kernel is

    :math:`K_{\theta}(X_0, X_1) = \tanh (\gamma X_0^\intercal \theta X_1 + r)`

    where :math:`\theta` is the kernel parameter, and :math:`\gamma` and r are
    scaling and offset coefficients."""

    gaussian_kernel_long_desc = r"""
    For tensors :math:`X_0` and :math:`X_1` containing features in column
    vectors, the parameterised Gaussian kernel is

    :math:`K_{\theta}(X_0, X_1) = e^{\frac{1}{\sigma^2} (X_0 - X_1)^\intercal \theta (X_0 - X_1)}`

    where :math:`\theta` is the kernel parameter, :math:`\sigma` is an
    isotropic standard deviation, and :math:`X_0 - X_1` contains all pairwise
    differences between vectors in :math:`X_0` and :math:`X_1`. The kernel
    parameter :math:`\theta` can also be interpreted as an inverse covariance.

    This is the same as :func:`rbf_kernel` but is parameterised in terms of
    :math:`\sigma` rather than  :math:`\gamma`."""

    rbf_kernel_long_desc = r"""
    For tensors :math:`X_0` and :math:`X_1` containing features in column
    vectors, the parameterised RBF kernel is

    :math:`K_{\theta}(X_0, X_1) = e^{\gamma (X_0 - X_1)^\intercal \theta (X_0 - X_1)}`

    where :math:`\theta` is the kernel parameter, :math:`\gamma` is a scaling
    coefficient, and :math:`X_0 - X_1` contains all pairwise differences
    between vectors in :math:`X_0` and :math:`X_1`. The kernel parameter
    :math:`\theta` can also be interpreted as an inverse covariance.

    This is the same as :func:`gaussian_kernel` but is parameterised in terms
    of :math:`\gamma` rather than  :math:`\sigma`."""

    cosine_kernel_long_desc = r"""
    For tensors :math:`X_0` and :math:`X_1` containing features in column
    vectors, the parameterised cosine kernel is

    :math:`K_{\theta}(X_0, X_1) = \frac{X_0^\intercal \theta X_1}{\|X_0\|_\theta \|X_1\|_\theta}`

    where the parameterised norm vector

    :math:`\|A\|_{\theta;i} = \sqrt{A_i^\intercal \theta A_i}`

    is the elementwise square root of the vector of quadratic forms."""

    norm_long_desc = r"""
    For a tensor :math:`X` containing features in column vectors, the
    parameterised norms of an observation vector :math:`X_i` are

    :math:`\|X_i\|_{\theta} = \sqrt{(X_i)^\intercal \theta (X_i)}`
    or
    :math:`\|X_i\|_{\theta} = (X_i)^\intercal \theta (X_i)`
    if ``squared`` is True. The non-squared form reduces to the L2 norm if
    ``theta`` is the identity matrix."""

    kernel_fmt_note = """
    .. note::
        The inputs here are assumed to contain features in row vectors and
        observations in columns. This differs from the convention frequently
        used in the literature. However, this has the benefit of direct
        compatibility with the top-k sparse tensor format."""

    pairwise_dim_spec = r"""
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
                    As above."""

    pairwise_base_spec = """
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
        used as the diagonal of the matrix."""

    kernel_gamma_spec = r"""
    gamma : float or None (default None)
        Scaling coefficient. If not explicitly specified, this is
        automatically set to :math:`\frac{1}{P}`."""

    kernel_sigma_spec = r"""
    sigma : float or None (default None)
        Isotropic standard deviation.
    """

    kernel_order_spec = """
    order : int (default 3)
        Maximum degree of the polynomial."""

    kernel_r_spec = r"""
    r : float (default 0)
        Offset coefficient."""

    kernel_return_spec = """
    Returns
    -------
    tensor
        Kernel Gram matrix."""

    fmt = NestedDocParse(
        linear_kernel_long_desc=linear_kernel_long_desc,
        polynomial_kernel_long_desc=polynomial_kernel_long_desc,
        sigmoid_kernel_long_desc=sigmoid_kernel_long_desc,
        gaussian_kernel_long_desc=gaussian_kernel_long_desc,
        rbf_kernel_long_desc=rbf_kernel_long_desc,
        cosine_kernel_long_desc=cosine_kernel_long_desc,
        norm_long_desc=norm_long_desc,
        kernel_fmt_note=kernel_fmt_note,
        pairwise_dim_spec=pairwise_dim_spec,
        pairwise_base_spec=pairwise_base_spec,
        kernel_gamma_spec=kernel_gamma_spec,
        kernel_sigma_spec=kernel_sigma_spec,
        kernel_order_spec=kernel_order_spec,
        kernel_r_spec=kernel_r_spec,
        kernel_return_spec=kernel_return_spec,
    )
    f.__doc__ = f.__doc__.format_map(fmt)
    return f


@singledispatch
@document_param_pairwise
def linear_kernel(
    X0: Tensor,
    X1: Optional[Tensor] = None,
    theta: Optional[Tensor] = None,
    intermediate_indices: None = None,
) -> Tensor:
    """
    Parameterised linear kernel between input tensors.
    \
    {linear_kernel_long_desc}
    \
    {kernel_fmt_note}
    \
    {pairwise_dim_spec}

    Parameters
    ----------\
    {pairwise_base_spec}
    \
    {kernel_return_spec}
    """
    if X1 is None:
        X1 = X0
    if theta is None:
        return X0 @ X1.swapaxes(-1, -2)
    elif theta.ndim == 1 or theta.shape[-1] != theta.shape[-2]:
        theta = _conform_vector_weight(theta)
        return (X0 * theta) @ X1.swapaxes(-1, -2)
    else:
        return X0 @ theta @ X1.swapaxes(-1, -2)


@linear_kernel.register
def _(
    X0: TopKTensor,
    X1: Optional[TopKTensor] = None,
    theta: Optional[Union[Tensor, TopKTensor]] = None,
    intermediate_indices: Union[None, Tensor, Tuple[Tensor, Tensor]] = None,
) -> TopKTensor:
    if X1 is None:
        X1 = X0
    if theta is None:
        return spspmm(X0, X1)
    elif is_sparse(theta):
        if theta.data.shape[-1] == theta.shape[-1]:
            lhs = X0
            if intermediate_indices is not None:
                rhs = topkx(spspmm)(intermediate_indices, X1, theta)
            else:
                rhs = spspmm(X1, theta)
        elif intermediate_indices is not None:
            mm = topkx(spspmm)
            lhs = mm(intermediate_indices[0], X0, theta)
            rhs = mm(intermediate_indices[1], X1, theta)
        else:
            lhs = spspmm(X0, theta)
            rhs = spspmm(X1, theta)
        return spspmm(lhs, rhs)
    elif theta.ndim == 1 or theta.shape[-1] != theta.shape[-2]:
        return spspmm(spdiagmm(X0, theta), X1)
    else:
        if intermediate_indices is not None:
            rhs = topkx(spspmm)(intermediate_indices, X1, theta)
            return spspmm(X0, rhs)
        return spspmm(X0, spspmm(X1, theta))


@singledispatch
@document_param_pairwise
def param_norm(
    X: Tensor,
    theta: Optional[Tensor] = None,
    *,
    squared: bool = False,
) -> Tensor:
    """
    Parameterised norm of observation vectors in an input tensor.
    \
    {norm_long_desc}
    \
    {kernel_fmt_note}
    """
    if theta is None:
        norm = jnp.linalg.norm(X, 2, axis=-1, keepdims=True)
        if squared:
            norm = norm**2
        Xnorm = X / norm
    elif theta.ndim == 1 or theta.shape[-1] != theta.shape[-2]:
        X = X[..., None, :]
        if theta.ndim > 1:
            theta = theta[..., None, None, :]
        norm = (X * theta) @ X.swapaxes(-1, -2)
        if not squared:
            norm = jnp.sqrt(norm)
        Xnorm = (X / norm).squeeze(-2)
    else:
        X = X[..., None, :]
        if theta.ndim > 2:
            theta = theta[..., None, :, :]
        norm = X @ theta @ X.swapaxes(-1, -2)
        if not squared:
            norm = jnp.sqrt(norm)
        Xnorm = (X / norm).squeeze(-2)

    return Xnorm


@param_norm.register
def _(
    X: TopKTensor,
    theta: Optional[Union[Tensor, TopKTensor]] = None,
    *,
    squared: bool = False,
) -> TopKTensor:
    lhs = X
    if theta is None:
        rhs = X
    elif theta.ndim == 1 or theta.shape[-1] != theta.shape[-2]:
        rhs = spdiagmm(X, theta)
    else:
        rhs = spspmm(X, theta)
    norm = spsp_innerpaired(lhs, rhs)[..., None]
    if not squared:
        norm = jnp.sqrt(norm)
    return BCOO(
        (X.data / norm, X.indices),
        shape=X.shape,
    )


@singledispatch
@document_param_pairwise
def linear_distance(
    X0: Tensor,
    X1: Optional[Tensor] = None,
    theta: Optional[Tensor] = None,
) -> Tensor:
    """
    Squared Euclidean (L2) distance (or Mahalanobis if theta is set).

    Parameters
    ----------\
    {pairwise_base_spec}

    Returns
    -------
    Tensor
        Distance matrix.
    """
    if X1 is None:
        X1 = X0
    D = X0[..., None, :] - X1[..., None, :, :]
    if (
        (theta is not None)
        and (theta.ndim > 1)
        and (theta.shape[-1] != theta.shape[-2])
    ):
        theta = theta[..., None, None, None, :]
    D = linear_kernel(D[..., None, :], theta=theta)
    return D.reshape(*D.shape[:-2])


@linear_distance.register
def _(
    X0: TopKTensor,
    X1: Optional[TopKTensor] = None,
    theta: Optional[Union[Tensor, TopKTensor]] = None,
) -> TopKTensor:
    if X1 is None:
        X1 = X0
    D = spsp_pairdiff(lhs=X0, rhs=X1)
    if isinstance(theta, BCOO):
        lhs = rhs = spspmm(D, theta)
    elif theta is not None:
        lhs = D
        if theta.ndim == 1 or theta.shape[-1] != theta.shape[-2]:
            rhs = spdiagmm(D, theta)
        else:
            theta = theta[..., None, :, :]
            rhs = spspmm(D, theta.swapaxes(-1, -2))
    else:
        lhs = rhs = D
    return spsp_innerpaired(lhs, rhs)


@document_param_pairwise
def polynomial_kernel(
    X0: Tensor,
    X1: Optional[Tensor] = None,
    theta: Optional[Tensor] = None,
    gamma: Optional[float] = None,
    order: int = 3,
    r: float = 0,
) -> Tensor:
    """
    Parameterised polynomial kernel between input tensors.
    \
    {polynomial_kernel_long_desc}
    \
    {kernel_fmt_note}
    \
    {pairwise_dim_spec}

    Parameters
    ----------\
    {pairwise_base_spec}\
    {kernel_gamma_spec}\
    {kernel_order_spec}\
    {kernel_r_spec}
    \
    {kernel_return_spec}
    """
    gamma = _default_gamma(X0, gamma=gamma)
    K = linear_kernel(X0, X1, theta)
    return (gamma * K + r) ** order


@document_param_pairwise
def sigmoid_kernel(
    X0: Tensor,
    X1: Optional[Tensor] = None,
    theta: Optional[Tensor] = None,
    gamma: Optional[float] = None,
    r: float = 0,
) -> Tensor:
    """
    Parameterised sigmoid kernel between input tensors.
    \
    {sigmoid_kernel_long_desc}
    \
    {kernel_fmt_note}
    \
    {pairwise_dim_spec}

    Parameters
    ----------\
    {pairwise_base_spec}\
    {kernel_gamma_spec}\
    {kernel_r_spec}
    \
    {kernel_return_spec}
    """
    gamma = _default_gamma(X0, gamma=gamma)
    K = linear_kernel(X0, X1, theta)
    return jax.nn.tanh(gamma * K + r)


@document_param_pairwise
def gaussian_kernel(
    X0: Tensor,
    X1: Optional[Tensor] = None,
    theta: Optional[Tensor] = None,
    sigma: Optional[float] = None,
) -> Tensor:
    """
    Parameterised Gaussian kernel between input tensors.
    \
    {gaussian_kernel_long_desc}
    \
    {kernel_fmt_note}
    \
    {pairwise_dim_spec}

    Parameters
    ----------\
    {pairwise_base_spec}\
    {kernel_sigma_spec}
    \
    {kernel_return_spec}
    """
    gamma = sigma
    if gamma is not None:
        gamma = sigma**-2
    return rbf_kernel(X0, X1, theta=theta, gamma=gamma)


@document_param_pairwise
def rbf_kernel(
    X0: Tensor,
    X1: Optional[Tensor] = None,
    theta: Optional[Tensor] = None,
    gamma: Optional[float] = None,
) -> Tensor:
    """
    Parameterised RBF kernel between input tensors.
    \
    {rbf_kernel_long_desc}
    \
    {kernel_fmt_note}
    \
    {pairwise_dim_spec}

    Parameters
    ----------\
    {pairwise_base_spec}\
    {kernel_gamma_spec}
    \
    {kernel_return_spec}
    """
    gamma = _default_gamma(X0, gamma=gamma)
    K = linear_distance(X0, X1, theta)
    return jnp.exp(-gamma * K)


@document_param_pairwise
def cosine_kernel(
    X0: Tensor,
    X1: Optional[Tensor] = None,
    theta: Optional[Tensor] = None,
) -> Tensor:
    """
    Parameterised cosine kernel between input tensors.
    \
    {cosine_kernel_long_desc}
    \
    {kernel_fmt_note}
    \
    {pairwise_dim_spec}

    Parameters
    ----------\
    {pairwise_base_spec}\
    {kernel_gamma_spec}
    \
    {kernel_return_spec}
    """
    X0_norm = param_norm(X0, theta=theta)
    if X1 is None:
        X1_norm = X0_norm
    else:
        X1_norm = param_norm(X1, theta=theta)
    return linear_kernel(X0_norm, X1_norm, theta)


@document_param_pairwise
def cov_kernel(
    X0: Tensor,
    X1: Optional[Tensor] = None,
    theta: Optional[Tensor] = None,
    l2: float = 0,
) -> Tensor:
    """
    Parameterised covariance kernel between input tensors.

    A thin wrapper around the :func:`pairedcov` function.
    \
    {pairwise_dim_spec}

    Parameters
    ----------\
    {pairwise_base_spec}
    \
    {kernel_return_spec}
    """
    if X1 is None:
        X1 = X0
    return pairedcov(X0, X1, weight=theta, l2=l2)


@document_param_pairwise
def corr_kernel(
    X0: Tensor,
    X1: Optional[Tensor] = None,
    theta: Optional[Tensor] = None,
    l2: float = 0,
) -> Tensor:
    """
    Parameterised correlation kernel between input tensors.

    A thin wrapper around the :func:`pairedcorr` function.
    \
    {pairwise_dim_spec}

    Parameters
    ----------\
    {pairwise_base_spec}
    \
    {kernel_return_spec}
    """
    if X1 is None:
        X1 = X0
    return pairedcorr(X0, X1, weight=theta, l2=l2)
