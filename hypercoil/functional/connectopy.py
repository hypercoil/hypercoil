# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Connectopic manifold mapping, based on ``brainspace``.
"""
import jax
import jax.numpy as jnp

from typing import Tuple
from hypercoil.functional.utils import Tensor, vmap_over_outer
from .graph import graph_laplacian
from .matrix import symmetric


def _absmax(X: Tensor) -> Tensor:
    # Not sure the reason for this way of implementing this in brainspace, but
    # I do believe they had a lot more time to evaluate different approaches,
    # so we'll go with theirs.
    return X[jnp.abs(X).argmax(axis=-2), jnp.arange(X.shape[-1])]


def _impose_sign_consistency(Q: Tensor) -> Tensor:
    """
    Consistent sign for eigenvectors, following ``brainspace`` convention.
    """
    sgn = jax.lax.stop_gradient(
        jnp.sign(vmap_over_outer(_absmax, 2)((Q,)))
    )
    return Q * sgn[..., None, :]


def laplacian_eigenmaps(
    W: Tensor,
    k: int = 10,
    normalise: bool = True,
) -> Tuple[Tensor, Tensor]:
    r"""
    Manifold coordinates estimated using Laplacian eigenmaps.

    .. warning::

        Sparse inputs are currently unsupported because an implementation of a
        sparse extremal eigenvalue solver does not yet exist in JAX. For
        sparse inputs, use the generalised connectopic functional instead --
        once we implement VJP rules for elementary operations on sparse
        matrices, anyway.

    :Dimension: **W :** :math:`(*, N, N)`
                    ``*`` denotes any number of preceding dimensions, N
                    denotes number of vertices, and E denotes number of edges.
                **Q :** :math:`(*, N, k)`
                    k denotes the number of eigenmaps.
                **L :** :math:`(*, k)`
                    As above.

    Parameters
    ----------
    W : tensor
        Edge weight tensor. If ``edge_index`` is not provided, then this
        should be the graph adjacency (or affinity) matrix; otherwise, it
        should be a list of weights corresponding to the edges in
        ``edge_index``.
    k : int (default 10)
        Number of eigenmaps to compute.
    normalise : bool (default True)
        Indicates that the Laplacian should be normalised using the degree
        matrix.

    Returns
    -------
    Q : tensor
        Eigenmaps.
    L : tensor
        Eigenvalues corresponding to eigenmaps.

    See also
    --------
    :func:`diffusion_mapping`
    """
    W = symmetric(W)
    H = graph_laplacian(W, normalise=normalise)
    #TODO: It would be great to derive an extremal eigenvalue solver that
    #      supports sparse matrices in the top-k format, imposing implicit
    #      symmetry. eigh is really much too inefficient for this to be
    #      currently practical.
    L, Q = jnp.linalg.eigh(H)
    L = L[..., 1:(k + 1)]
    Q = Q[..., 1:(k + 1)]

    if normalise:
        D = jnp.sqrt(W.sum(-1, keepdims=True))
        Q = Q / D

    Q = _impose_sign_consistency(Q)
    return Q, L


def diffusion_mapping(
    W: Tensor,
    k: int = 10,
    alpha: float = 0.5,
    diffusion_time: int = 0,
) -> Tuple[Tensor, Tensor]:
    r"""
    Manifold coordinates estimated using diffusion mapping.

    This functionality is adapted very closely from ``brainspace`` with some
    minor adaptations for differentiability.

    .. warning::

        Sparse inputs are currently unsupported because an implementation of a
        sparse extremal eigenvalue solver does not yet exist in JAX. For
        sparse inputs, use the generalised connectopic functional instead --
        once we implement VJP rules for elementary operations on sparse
        matrices, anyway.

    .. note::

        The anisotropic diffusion parameter determines the kind of diffusion
        map produced by the algorithm.

        * :math:`\alpha = 0` produces Laplacian eigenmaps, corresponding to a
          random walk-style diffusion operator.
        * :math:`\alpha = 0.5` (default) corresponds to Fokker-Planck
          diffusion.
        * :math:`\alpha = 1` corresponds to Laplace-Beltrami diffusion.

    :Dimension: **W :** :math:`(*, N, N)` or :math:`(*, E)`
                    ``*`` denotes any number of preceding dimensions, N
                    denotes number of vertices, and E denotes number of edges.
                    The shape should be :math:`(*, N, N)` if ``edge_index`` is
                    not provided and :math:`(*, E)` if ``edge_index`` is
                    provided.
                **edge_index :** :math:`(*, 2, E)`
                    As above.
                **Q :** :math:`(*, N, k)`
                    k denotes the number of diffusion maps.
                **L :** :math:`(*, k)`
                    As above.

    Parameters
    ----------
    W : tensor
        Edge weight tensor. This should be the graph adjacency (or affinity)
        matrix.
    k : int (default 10)
        Number of eigenmaps to compute.
    alpha : float :math:`\in [0, 1]` (default 0.5)
        Anisotropic diffusion parameter.
    diffusion_time : int (default 0)
        Diffusion time parameter. A value of 0 indicates that a multi-scale
        diffusion map should be computed, which considers all valid times
        (1, 2, 3, etc.).

    Returns
    -------
    Q : tensor
        Diffusion maps.
    L : tensor
        Eigenvalues corresponding to diffusion maps.

    See also
    --------
    :func:`laplacian_eigenmaps`
    """
    W = symmetric(W)

    if alpha > 0:
        D = W.sum(axis=-1, keepdims=True)
        D_power = D ** -alpha
        W = D_power * W * D_power.swapaxes(-1, -2)

    D = W.sum(axis=-1, keepdims=True)
    W = W * (D ** -1)

    #TODO: It would be great to derive an extremal eigenvalue solver that
    #      supports sparse matrices in the top-k format, imposing implicit
    #      symmetry. eigh is really much too inefficient for this to be
    #      currently practical.
    L, Q = jnp.linalg.eigh(W)
    L = jnp.flip(L[..., -(k + 1):], -1)
    Q = jnp.flip(Q[..., -(k + 1):], -1)

    L = L / L[..., 0]
    Q = Q / Q[..., [0]]
    L = L[..., 1:]
    Q = Q[..., 1:]

    if diffusion_time <= 0:
        L = L / (1 - L)
    else:
        L = L ** diffusion_time

    Q = Q * L[..., None, :]
    Q = _impose_sign_consistency(Q)
    return Q, L
