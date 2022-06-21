# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Connectopic gradients
~~~~~~~~~~~~~~~~~~~~~
Connectopic gradient mapping, based on brainspace.
"""
import torch
from .graph import graph_laplacian
from .matrix import symmetric, symmetric_sparse


def laplacian_eigenmaps(W, edge_index=None, k=10, normalise=True):
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
    return L, Q
