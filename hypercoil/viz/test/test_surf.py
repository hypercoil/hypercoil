# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for surfplot-based visualisations
"""
import pytest

from pkg_resources import resource_filename as pkgrf

import jax
import templateflow.api as tflow
from neuromaps.transforms import mni152_to_fsaverage

from hypercoil.init.atlas import DirichletInitSurfaceAtlas
from hypercoil.viz.surf import (
    CortexTriSurface,
    make_cmap,
)
from hypercoil.viz.surfplot import plot_surf_scalars
from hypercoil.viz.utils import plot_to_image
from hypercoil.viz.flows import (
    ichain,
    ochain,
    iochain,
    map_over_sequence,
    split_chain,
    apply_along_axis,
    replicate,
)
from hypercoil.viz.transforms import (
    surf_from_archive,
    resample_to_surface,
    plot_and_save,
    scalars_from_cifti,
    scalars_from_atlas,
    parcellate_colormap,
    row_major_grid,
    col_major_grid,
    save_fig,
    select_closest_poles,
)


class TestSurfaceVisualisations:
    @pytest.mark.ci_unsupported
    def test_parcellation(self):
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
    def test_scalar(self):
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
    def test_continuous_parcellation(self):
        i_chain = ichain(
            surf_from_archive(),
            resample_to_surface('difumo', template='fsaverage'),
            replicate(mapping={"scalars": [f"difumo_{i}" for i in range(64)]}),
            select_closest_poles(projection='pial', n_poles=3),
        )
        o_chain = ochain(
            row_major_grid(ncol=3, figsize=(3, 10), num_panels=30),
            save_fig()
        )
        f = iochain(plot_surf_scalars, i_chain, o_chain)
        nii = tflow.get(
            template='MNI152NLin2009cAsym',
            atlas='DiFuMo',
            resolution=2,
            desc='64dimensions'
        )
        f(
            template="fsaverage",
            load_mask=True,
            nii=nii,
            projection='pial',
            filename='/tmp/parcelpeaks_index-{index}.png',
            clim=(0, 2e-4),
            cmap='magma',
            window_size=(400, 250),
        )

    @pytest.mark.ci_unsupported
    def test_parcellation_from_atlas(self):
        i_chain = ichain(
            surf_from_archive(),
            scalars_from_atlas('dirichlet'),
            replicate(mapping={"scalars": [f"dirichlet_{i}" for i in range(40)], "hemi": ["left" if i < 20 else "right" for i in range(40)]}),
            select_closest_poles(projection='veryinflated', n_poles=3),
        )
        o_chain = ochain(
            row_major_grid(ncol=3, figsize=(3, 10), num_panels=30),
            save_fig()
        )
        f = iochain(plot_surf_scalars, i_chain, o_chain)

        cifti_template = pkgrf(
            'hypercoil',
            'viz/resources/nullexample.nii'
        )
        atlas = DirichletInitSurfaceAtlas(
            cifti_template=cifti_template,
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
            compartment_labels={
                'cortex_L': 20,
                'cortex_R': 20,
                'subcortex': 0,
            },
            key=jax.random.PRNGKey(0),
        )
        f(
            template="fsLR",
            load_mask=True,
            atlas=atlas,
            projection='veryinflated',
            filename='/tmp/dirpeaks_index-{index}.png',
            clim=(0.045, 0.055)
        )
        assert 0
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
