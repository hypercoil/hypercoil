# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Geometric compression
~~~~~~~~~~~~~~~~~~~~~
Vertical compression groupings based on geometric adjacencies.
"""
import torch
import numpy as np
from scipy.linalg import toeplitz
from scipy.sparse import coo_matrix, eye


def construct_adjacency_matrix(n_vertices_in, edges, directed=False):
    I = np.empty(n_vertices_in + len(edges), dtype='long')
    J = np.empty(n_vertices_in + len(edges), dtype='long')
    I[:n_vertices_in] = np.arange(n_vertices_in)
    J[:n_vertices_in] = np.arange(n_vertices_in)
    I[n_vertices_in:] = [i for i, _ in edges]
    J[n_vertices_in:] = [j for _, j in edges]
    data = np.ones_like(I, dtype='bool')
    T = coo_matrix((data, (I, J)), shape=(n_vertices_in, n_vertices_in))
    if directed:
        T = T.astype('bool')
    else:
        T = (T + T.T).astype('bool')
    return T


def construct_group_matrices(n_groups, n_vertices_in):
    group_matrix = {}
    n_vertices_out = int(np.ceil(n_vertices_in / n_groups))
    for g in range(n_groups):
        i = np.arange(n_vertices_out, dtype='long')
        j = np.arange(start=0, stop=n_vertices_in, step=n_groups,
                      dtype='long') + g
        if j[-1] >= n_vertices_in:
            i = i[:-1]
            j = j[:-1]
        data = np.ones_like(i, dtype='bool')
        group_matrix[g] = coo_matrix(
            (data, (i, j)),
            shape=(n_vertices_out, n_vertices_in),
            dtype='bool'
        )
    return group_matrix


def compression_matrix(adjmat, walk_weights, group_matrix):
    max_walk = len(walk_weights)

    walks = [None for _ in range(max_walk)]
    masked_walks = [None for _ in range(max_walk)]
    power = eye(adjmat.shape[0], dtype='bool')
    prev = 0

    for s in range(max_walk):
        w = power @ group_matrix
        walks[s] = w
        masked_walks[s] = w - prev
        power = adjmat @ power
        prev = prev + w
        
    return sum([
        wei * walk.astype('int')
        for wei, walk in zip(walk_weights, masked_walks)
    ])


def edges_from_tri_mesh(mesh):
    edges = []
    for tri in mesh:
        x, y, z = tri
        edges += [
            (x, y), (x, z), (y, z)
        ]
    return set(edges)


def compressions_from_gifti(path):
    surf = nb.load(path)
    n_vertices_in = surf.darrays[0].data.shape[0]
    edges = edges_from_tri_mesh(surf.darrays[1].data)
    T = construct_adjacency_matrix(edges=edges, n_vertices_in=n_vertices_in)
    group_matrix = construct_group_matrices(
        n_groups=n_groups,
        n_vertices_in=n_vertices_in
    )
    return [compression_matrix(
        adjmat=T,
        walk_weights=walk_weights,
        group_matrix=m.T
    ) for m in group_matrix.values()]


def compression_block_tensor(matrices, device=None, dtype=None):
    i = np.zeros((3, 0))
    v = np.zeros(0)
    for g, matrix in enumerate(matrices):
        matrix = matrix.tocoo()
        r = matrix.row
        c = matrix.col
        v = np.concatenate((v, matrix.data))
        j = g * np.ones_like(r)
        i = np.concatenate((i, np.stack((j, r, c))), axis=-1)
    return torch.sparse_coo_tensor(i, v, (n_groups, *matrix.shape),
                                   device=device, dtype=dtype)
