# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for elementary surfplot-based visualisations
"""
import pytest

from pkg_resources import resource_filename as pkgrf

import templateflow.api as tflow

from hypercoil.viz.surfplot import plot_surf_scalars
from hypercoil.viz.flows import (
    ichain,
    ochain,
    iochain,
    map_over_sequence,
    split_chain,
)
from hypercoil.viz.transforms import (
    surf_from_archive,
    resample_to_surface,
    plot_and_save,
    scalars_from_cifti,
    parcellate_colormap,
)


class TestSurfaceVisualisations:
    @pytest.mark.ci_unsupported
    def test_scalars(self):
        i_chain = ichain(
            surf_from_archive(),
            resample_to_surface('gm_density', template='fsaverage'),
        )
        o_chain = ochain(
            map_over_sequence(
                xfm=plot_and_save(),
                mapping={
                    "basename": ('/tmp/left_density', '/tmp/right_density'),
                    "hemi": ('left', 'right'),
                }
            )
        )
        f = iochain(plot_surf_scalars, i_chain, o_chain)
        out = f(
            template="fsaverage",
            load_mask=True,
            nii=tflow.get(
                template='MNI152NLin2009cAsym',
                suffix='probseg',
                label="GM",
                resolution=2
            ),
            projection='pial',
            scalars='gm_density',
        )
        assert len(out.keys()) == 1
        assert "screenshots" in out.keys()

    @pytest.mark.ci_unsupported
    def test_parcellation(self):
        i_chain = ichain(
            surf_from_archive(),
            scalars_from_cifti('parcellation'),
            parcellate_colormap('network', 'parcellation')
        )
        o_chain = ochain(
            split_chain(
                map_over_sequence(
                    xfm=plot_and_save(),
                    mapping={
                        "basename": ('/tmp/left', '/tmp/right'),
                        "hemi": ('left', 'right'),
                    }
                ),
                map_over_sequence(
                    xfm=plot_and_save(),
                    mapping={
                        "basename": ('/tmp/left', '/tmp/right'),
                        "hemi": ('left', 'right'),
                        "views": (((-20, 0, 0),), (((65, 65, 0), (0, 0, 0), (0, 0, 1)),))
                    }
                ),
            )
        )
        f = iochain(plot_surf_scalars, i_chain, o_chain)
        f(
            template="fsLR",
            load_mask=True,
            cifti=pkgrf(
                'hypercoil',
                'viz/resources/nullexample.nii'
            ),
            projection='veryinflated',
            scalars='parcellation',
            boundary_color='black',
            boundary_width=5,
        )

    @pytest.mark.ci_unsupported
    def test_parcellation_modal_cmap(self):
        i_chain = ichain(
            surf_from_archive(),
            scalars_from_cifti('parcellation'),
            parcellate_colormap('modal', 'parcellation')
        )
        o_chain = ochain(
            split_chain(
                map_over_sequence(
                    xfm=plot_and_save(),
                    mapping={
                        "basename": ('/tmp/leftmodal', '/tmp/rightmodal'),
                        "hemi": ('left', 'right'),
                    }
                ),
                map_over_sequence(
                    xfm=plot_and_save(),
                    mapping={
                        "basename": ('/tmp/leftmodal', '/tmp/rightmodal'),
                        "hemi": ('left', 'right'),
                        "views": (((-20, 0, 0),), (((65, 65, 0), (0, 0, 0), (0, 0, 1)),))
                    }
                ),
            )
        )
        f = iochain(plot_surf_scalars, i_chain, o_chain)
        f(
            template="fsLR",
            load_mask=True,
            cifti=pkgrf(
                'hypercoil',
                'viz/resources/nullexample.nii'
            ),
            projection='veryinflated',
            scalars='parcellation',
            boundary_color='black',
            boundary_width=5,
        )

    # @pytest.fixture(autouse=True)
    # def setup_class(self):
    #     ref_pointer = pkgrf(
    #         'hypercoil',
    #         'viz/resources/nullexample.nii'
    #     )
    #     self.atlas = CortexSubcortexCIfTIAtlas(
    #         ref_pointer=ref_pointer,
    #         mask_L=tflow.get(
    #             template='fsLR',
    #             hemi='L',
    #             desc='nomedialwall',
    #             density='32k'),
    #         mask_R=tflow.get(
    #             template='fsLR',
    #             hemi='R',
    #             desc='nomedialwall',
    #             density='32k'),
    #         clear_cache=False,
    #         dtype=torch.float
    #     )
    #     self.lin = AtlasLinear(self.atlas)
    #     self.modal_cmap = pkgrf(
    #         'hypercoil',
    #         'viz/resources/cmap_modal.nii'
    #     )
    #     self.network_cmap = pkgrf(
    #         'hypercoil',
    #         'viz/resources/cmap_network.nii'
    #     )

    # @pytest.mark.ci_unsupported
    # def test_parcellation_plotter(self):
    #     all_views = (
    #         'dorsal', 'ventral',
    #         'posterior', 'anterior',
    #         'medial', 'lateral'
    #     )
    #     results = pkgrf(
    #         'hypercoil',
    #         'results/'
    #     )
    #     plotter = fsLRAtlasParcels(self.lin)
    #     plotter(
    #         cmap=self.modal_cmap,
    #         views=all_views,
    #         save=f'{results}/parcellation_cmap-modal'
    #     )
    #     plotter(
    #         cmap=self.network_cmap,
    #         views=all_views,
    #         save=f'{results}/parcellation_cmap-network'
    #     )
    #     plotter(
    #         cmap='RdBu_r',
    #         views=all_views,
    #         scores=torch.randn(400),
    #         save=f'{results}/parcellation_desc-randomstats'
    #     )

    # @pytest.mark.ci_unsupported
    # def test_map_plotter(self):
    #     results = pkgrf(
    #         'hypercoil',
    #         'results/'
    #     )
    #     plotter = fsLRAtlasMaps(self.lin)
    #     plotter(save=f'{results}/parcellation_maps', stop_batch=1)
    #     plotter(save=f'{results}/parcellation_maps_selected',
    #             select_nodes=[1, 3, 9, 221, 235])
