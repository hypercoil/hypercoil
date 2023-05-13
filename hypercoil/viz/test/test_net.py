# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for netplot-based visualisations
"""
import pytest
import numpy as np
import pandas as pd

from pkg_resources import resource_filename as pkgrf

from hypercoil.viz.surf import (
    CortexTriSurface,
    make_cmap,
)
from hypercoil.viz.netplot import (
    plot_embedded_graph,
    filter_adjacency_data,
)
from hypercoil.viz.utils import plot_to_image


class TestNetworkVisualisations:
    @pytest.mark.ci_unsupported
    def test_net(self):
        surf = CortexTriSurface.from_tflow(
            template="fsLR",
            load_mask=True,
            projections=('inflated',)
        )
        parcellation = '/Users/rastkociric/Downloads/desc-schaefer_res-0400_atlas.nii'
        cov = pd.read_csv(pkgrf(
            'hypercoil',
            'examples/synthetic/data/synth-regts/'
            f'atlas-schaefer400_desc-synth_cov.tsv'
        ), sep='\t', header=None).values
        node_values = np.maximum(cov, 0).sum(axis=0)
        node_values = pd.DataFrame(
            node_values,
            index=range(1, 401),
            columns=('degree',)
        )
        node_edge_selection = np.zeros(cov.shape[0], dtype=bool)
        node_edge_selection[0:5] = True
        node_edge_selection[200:205] = True
        edge_values = filter_adjacency_data(
            cov,
            threshold=10,
            topk_threshold_nodewise=True,
            absolute_threshold=True,
            node_selection=node_edge_selection,
        )
        surf.add_cifti_dataset(
            name='parcellation',
            cifti=parcellation,
            is_masked=True,
            apply_mask=False,
            null_value=None,
        )
        cmap = pkgrf(
            'hypercoil',
            'viz/resources/cmap_network.nii'
        )
        surf.add_cifti_dataset(
            name='cmap',
            cifti=cmap,
            is_masked=True,
            apply_mask=False,
            null_value=0.
        )
        node_lh = np.arange(400) < 200
        cmap, clim = make_cmap(
            surf, 'cmap', 'parcellation', separate=False)
        p = plot_embedded_graph(
            surf=surf,
            edge_values=edge_values,
            node_values=node_values,
            node_lh=node_lh,
            parcellation=parcellation,
            projection='inflated',
            node_radius='degree',
            node_color='index',
            node_cmap=cmap,
        )
        plot_to_image(
            p, basename='/tmp/net',
            views=("dorsal", "left", "posterior"),
            hemi='both'
        )
