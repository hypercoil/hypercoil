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

from hypercoil.viz.netplot import (
    plot_embedded_graph,
)
from hypercoil.viz.flows import (
    ichain,
    ochain,
    iochain,
    joindata,
)
from hypercoil.viz.transforms import (
    surf_from_archive,
    plot_and_save,
    parcellate_colormap,
    scalars_from_cifti,
    add_edge_variable,
    add_node_variable,
)


class TestNetworkVisualisations:
    @pytest.mark.ci_unsupported
    def test_net(self):
        parcellation = '/Users/rastkociric/Downloads/desc-schaefer_res-0400_atlas.nii'
        cov = pd.read_csv(pkgrf(
            'hypercoil',
            'examples/synthetic/data/synth-regts/'
            f'atlas-schaefer400_desc-synth_cov.tsv'
        ), sep='\t', header=None).values

        vis_nodes_edge_selection = np.zeros(400, dtype=bool)
        vis_nodes_edge_selection[0:5] = True
        vis_nodes_edge_selection[200:205] = True

        i_chain = ichain(
            surf_from_archive(),
            joindata(fill_value=0.)(
                add_edge_variable(
                    "vis_conn",
                    threshold=10,
                    topk_threshold_nodewise=True,
                    absolute=True,
                    incident_node_selection=vis_nodes_edge_selection,
                    emit_degree=True,
                ),
                add_edge_variable(
                    "vis_internal_conn",
                    absolute=True,
                    connected_node_selection=vis_nodes_edge_selection,
                ),
            ),
            scalars_from_cifti('parcellation'),
            parcellate_colormap('network', 'parcellation'),
        )
        o_chain = ochain(
            plot_and_save(),
        )
        f = iochain(plot_embedded_graph, i_chain, o_chain)
        f(
            template="fsLR",
            node_lh=(np.arange(400) < 200),
            cifti=parcellation,
            parcellation='parcellation',
            projection='inflated',
            node_radius='vis_conn_degree',
            node_color='index',
            edge_color='vis_conn_sgn',
            edge_radius='vis_conn_val',
            vis_conn_adjacency=cov,
            vis_internal_conn_adjacency=cov,
            basename='/tmp/net',
            views=("dorsal", "left", "posterior"),
            hemi='both'
        )

    def test_net_highlight(self):
        parcellation = '/Users/rastkociric/Downloads/desc-schaefer_res-0400_atlas.nii'
        cov = pd.read_csv(pkgrf(
            'hypercoil',
            'examples/synthetic/data/synth-regts/'
            f'atlas-schaefer400_desc-synth_cov.tsv'
        ), sep='\t', header=None).values

        vis_nodes_edge_selection = np.zeros(400, dtype=bool)
        vis_nodes_edge_selection[0:2] = True
        vis_nodes_edge_selection[200:202] = True

        i_chain = ichain(
            surf_from_archive(),
            joindata(fill_value=0., how="left")(
                add_edge_variable(
                    "vis_conn",
                    absolute=True,
                    incident_node_selection=vis_nodes_edge_selection,
                ),
                add_edge_variable(
                    "vis_internal_conn",
                    absolute=True,
                    connected_node_selection=vis_nodes_edge_selection,
                    emit_degree=True,
                    emit_incident_nodes=(0.1, 1),
                    removed_val=0.05,
                    surviving_val=1.0,
                ),
            ),
            scalars_from_cifti('parcellation'),
            parcellate_colormap('modal', 'parcellation'),
        )
        o_chain = ochain(
            plot_and_save(),
        )
        f = iochain(plot_embedded_graph, i_chain, o_chain)
        f(
            template="fsLR",
            node_lh=(np.arange(400) < 200),
            cifti=parcellation,
            parcellation='parcellation',
            projection='inflated',
            node_radius='vis_internal_conn_degree',
            node_color='index',
            node_opacity='vis_internal_conn_incidents',
            edge_color='vis_conn_sgn',
            edge_radius='vis_conn_val',
            edge_opacity='vis_internal_conn_val',
            vis_conn_adjacency=cov,
            vis_internal_conn_adjacency=cov,
            basename='/tmp/nethighlight',
            views=("dorsal", "left", "posterior"),
            hemi='both'
        )
