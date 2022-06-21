# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Connectopic gradient mapping, based on brainspace.
"""
import torch
from .graph import graph_laplacian
from .matrix import symmetric, symmetric_sparse


def laplacian_eigenmaps(W, edge_index=None, k=10, normalise=True):
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

    Returns
    -------
    Q : tensor
        Eigenmaps.
    L : tensor
        Eigenvalues corresponding to eigenmaps.
    """
    if edge_index is None:
        W = symmetric(W)
    else:
        W, edge_index = symmetric_sparse(W, edge_index)
    H = graph_laplacian(W, edge_index=edge_index, normalise=normalise)
    L, Q = torch.lobpcg(
        A=H,
        #B=D,
        k=(k + 1),
        largest=False
    )
    L = L[..., 1:]
    Q = Q[..., 1:]
    if normalise:
        D = W.sum(-1, keepdim=True).sqrt()
        Q = Q / D

    # Consistent sign, following brainspace convention.
    Q = Q * torch.sign(Q[Q.abs().argmax(0), range(k)])
    return Q, L
