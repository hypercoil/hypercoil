# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Connectopic gradient mapping, based on ``brainspace``.
"""
import torch
from math import sqrt
from .graph import graph_laplacian
from .matrix import symmetric, symmetric_sparse


def _symmetrise(W, edge_index, return_coo=False):
    if edge_index is None:
        W = symmetric(W)
    elif return_coo:
        return symmetric_sparse(
            W, edge_index,
            divide=False,
            return_coo=True
        ), None
    else:
        W, edge_index = symmetric_sparse(W, edge_index, divide=False)
    return W, edge_index


def _impose_sign_consistency(Q, k):
    """
    Consistent sign for eigenvectors, following ``brainspace`` convention.
    """
    with torch.no_grad():
        sgn = torch.sign(Q[Q.abs().argmax(0), range(k)])
    return Q * sgn


def laplacian_eigenmaps(W, edge_index=None, k=10,
                        normalise=True, method='lobpcg'):
    """
    Manifold coordinates estimated using Laplacian eigenmaps.

    :Dimension: **W :** :math:`(*, N, N)` or :math:`(*, E)`
                    ``*`` denotes any number of preceding dimensions, N
                    denotes number of vertices, and E denotes number of edges.
                    The shape should be :math:`(*, N, N)` if ``edge_index`` is
                    not provided and :math:`(*, E)` if ``edge_index`` is
                    provided.
                **edge_index :** :math:`(*, 2, E)`
                    As above.
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
    edge_index : ``LongTensor`` or None (default None)
        List of edges corresponding to the provided weights. Each column
        contains the index of the source vertex and the index of the target
        vertex for the corresponding weight in ``W``.
    k : int (default 10)
        Number of eigenmaps to compute.
    normalise : bool (default True)
        Indicates that the Laplacian should be normalised using the degree
        matrix.
    method : ``'lobpcg'`` (default) or ``'eigh'``
        Method for computing the eigendecomposition.

    Returns
    -------
    Q : tensor
        Eigenmaps.
    L : tensor
        Eigenvalues corresponding to eigenmaps.
    """
    W, edge_index = _symmetrise(W, edge_index)
    H = graph_laplacian(W, edge_index=edge_index, normalise=normalise)
    if isinstance(H, tuple):
        H = torch.sparse_coo_tensor(
            indices=H[0],
            values=H[1]
        )
    #TODO: LOBPCG is currently not as efficient as it could be. See:
    # https://github.com/pytorch/pytorch/issues/58828
    # and monitor progress.
    # https://github.com/rfeinman/Torch-ARPACK relevant but looks dead,
    # and we need multiple extremal eigenpairs.
    if edge_index is not None or method == 'lobpcg':
        L, Q = torch.lobpcg(
            A=H,
            #B=D,
            k=(k + 1),
            largest=False
        )
        L = L[..., 1:]
        Q = Q[..., 1:]
    else:
        L, Q = torch.linalg.eigh(H)
        L = L[..., 1:(k + 1)]
        Q = Q[..., 1:(k + 1)]

    if normalise:
        D = W.sum(-1, keepdim=True).sqrt()
        Q = Q / D

    Q = _impose_sign_consistency(Q, k)
    return Q, L


def diffusion_mapping(W, edge_index=None, k=10, alpha=0.5,
                      diffusion_time=0, method='lobpcg', niter_svd=500):
    """
    This is adapted very closely from ``brainspace``.
    method : ``'lobpcg'`` (default) or ``'eigh'`` or ``'svd'``
        Method for computing the eigendecomposition.
    """
    W, _ = _symmetrise(W, edge_index, return_coo=True)

    if alpha > 0:
        if edge_index is not None:
            D = torch.sparse.sum(W, 1).to_dense()
            D_power = D ** -alpha
            # Note that the first two dimensions contain sparse matrix slices,
            # and the last dimension is batch if applicable.
            row, col = W.indices()
            values = (
                D_power[row] *
                W.values() *
                D_power[col]
            )
            W = torch.sparse_coo_tensor(
                indices=W.indices(),
                values=values
            ).coalesce()
        else:
            D = W.sum(axis=-1, keepdim=True)
            D_power = D ** -alpha
            W = D_power * W * D_power.transpose(-1, -2)

    if edge_index is not None:
        D = torch.sparse.sum(W, 1).to_dense()
        D_power = D[W.indices()[0]] ** -1
        W = torch.sparse_coo_tensor(
            indices=W.indices(),
            values=W.values() * D_power
        ).coalesce()
    else:
        D = W.sum(axis=-1, keepdim=True)
        W = W * (D ** -1)

    #TODO: LOBPCG is currently not as efficient as it could be. See:
    # https://github.com/pytorch/pytorch/issues/58828
    # and monitor progress.
    # https://github.com/rfeinman/Torch-ARPACK relevant but looks dead,
    # and we need multiple extremal eigenpairs.
    if method == 'lobpcg':
        L, Q = torch.lobpcg(
            A=W,
            k=(k + 1),
            largest=True
        )
        print(Q, L)
        print(Q.shape, L.shape)
    elif method == 'eigh':
        L, Q = torch.linalg.eigh(W)
        L = L[..., -(k + 1):].flip(-1)
        Q = Q[..., -(k + 1):].flip(-1)
    elif method == 'svd':
        Q, L, _ = torch.svd_lowrank(W, q=(k + 1), niter=niter_svd)
        print(Q, L)
        print(Q.shape, L.shape)
        #assert 0

    L = L / L[..., 0]
    Q = Q / Q[..., [0]]
    L = L[..., 1:]
    Q = Q[..., 1:]

    if diffusion_time <= 0:
        L = L / (1 - L)
    else:
        L = L ** diffusion_time

    Q = Q * L.unsqueeze(-2)
    Q = _impose_sign_consistency(Q, k)
    return Q, L
