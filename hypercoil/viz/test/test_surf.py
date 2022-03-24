# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for surfplot-based visualisations
"""
import pytest
import torch
from templateflow import api as tflow
from pkg_resources import resource_filename as pkgrf
from hypercoil.init.atlas import CortexSubcortexCIfTIAtlas
from hypercoil.nn import AtlasLinear
from hypercoil.viz.surf import (
    fsLRAtlasParcels,
    fsLRAtlasMaps
)


class TestSurfaceVisualisations:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        #TODO: These hard codes really need to be changed at some point.
        # It's just too bad that the references we're testing with are
        # barred from redistribution.
        self.atlas = CortexSubcortexCIfTIAtlas(
            ref_pointer='/Users/rastkociric/Downloads/glasser.nii',
            mask_L=tflow.get(
                template='fsLR',
                hemi='L',
                desc='nomedialwall',
                density='32k'),
            mask_R=tflow.get(
                template='fsLR',
                hemi='R',
                desc='nomedialwall',
                density='32k'),
            clear_cache=False,
            dtype=torch.float
        )
        self.lin = lin = AtlasLinear(self.atlas)
        self.modal_cmap = pkgrf(
            'hypercoil',
            'viz/resources/cmap_modal.nii'
        )
        self.network_cmap = pkgrf(
            'hypercoil',
            'viz/resources/cmap_network.nii'
        )

    def test_parcellation_plotter(self):
        all_views = (
            'dorsal', 'ventral',
            'posterior', 'anterior',
            'medial', 'lateral'
        )
        results = pkgrf(
            'hypercoil',
            'results/'
        )
        plotter = fsLRAtlasParcels(self.lin)
        plotter(
            cmap=self.modal_cmap,
            views=all_views,
            save=f'{results}/parcellation_cmap-modal'
        )
        plotter(
            cmap=self.network_cmap,
            views=all_views,
            save=f'{results}/parcellation_cmap-network'
        )

    def test_map_plotter(self):
        results = pkgrf(
            'hypercoil',
            'results/'
        )
        plotter = fsLRAtlasMaps(self.lin)
        plotter(save=f'{results}/parcellation_maps', stop_batch=1)