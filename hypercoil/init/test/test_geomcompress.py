# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for geometric vertical compressions
"""
import pytest
import numpy as np
import nibabel as nb
import templateflow.api as tflow
from torch import half
from hypercoil.init.geomcompress import (
    construct_adjacency_matrix,
    construct_group_matrices,
    compression_matrix,
    compressions_from_gifti,
    compression_block_tensor,
    edges_from_tri_mesh
)


class TestGeometricVerticalCompression:

    def test_small(self):
        n_vertices_in = 10
        n_groups = 2
        edges = [
            (1, 7), (3, 4), (9, 2), (6, 3),
            (8, 2), (1, 4), (1, 9), (0, 4)
        ]
        walk_weights = [1, 0.25, 0.05, 0.01, 0.001]
        T = construct_adjacency_matrix(n_vertices_in, edges)
        group_matrix = construct_group_matrices(
            n_groups=n_groups,
            n_vertices_in=n_vertices_in
        )
        C0 = compression_matrix(
            adjmat=T,
            walk_weights=walk_weights,
            group_matrix=group_matrix[0].T
        )
        C1 = compression_matrix(
            adjmat=T,
            walk_weights=walk_weights,
            group_matrix=group_matrix[1].T
        )

        ref = np.array([
            [1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 1., 0., 0., 1., 0., 1.],
            [0., 0., 1., 0., 0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 1., 1., 0., 1., 0., 0., 0.],
            [1., 1., 0., 1., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 1., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 1., 0.],
            [0., 1., 1., 0., 0., 0., 0., 0., 0., 1.]
        ])
        assert np.all(T.toarray() == ref)

        ref = np.array([
            [1.0000, 0.0500, 0.0010, 0.0500, 0.2500,
             0.0000, 0.0100, 0.0100, 0.0000, 0.0100],
            [0.0010, 0.0500, 1.0000, 0.0010, 0.0100,
             0.0000, 0.0000, 0.0100, 0.2500, 0.2500],
            [0.2500, 0.2500, 0.0100, 0.2500, 1.0000,
             0.0000, 0.0500, 0.0500, 0.0010, 0.0500],
            [0.0100, 0.0100, 0.0000, 0.2500, 0.0500,
             0.0000, 1.0000, 0.0010, 0.0000, 0.0010],
            [0.0000, 0.0100, 0.2500, 0.0000, 0.0010,
             0.0000, 0.0000, 0.0010, 1.0000, 0.0500]
        ]).T
        assert np.all(C0 == ref)

    def test_gifti(self):
        n_groups = 10
        walk_weights = [1, 0.25, 0.05, 0.01, 0.001]
        surf_path = tflow.get(
            template='fsLR',
            space=None,
            density='32k',
            suffix='sphere',
            hemi='L'
        )
        surf = nb.load(surf_path).darrays[1].data
        edges = edges_from_tri_mesh(surf)
        CCC = compressions_from_gifti(
            path=surf_path,
            n_groups=n_groups,
            walk_weights=walk_weights
        )
        CCCT = compression_block_tensor(CCC, dtype=half)
        for (i, j) in list(edges)[:100]:
            print(i, j)
            k = i % n_groups
            i = i // n_groups
            assert CCC[k][j, i] == CCCT[k, j, i]
            assert CCCT[k, j, i] >= 0.25
