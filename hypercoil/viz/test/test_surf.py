# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for surfplot-based visualisations
"""
import pytest

from pkg_resources import resource_filename as pkgrf

import templateflow.api as tflow
from neuromaps.transforms import mni152_to_fsaverage

from hypercoil.viz.surf import (
    CortexTriSurface,
    make_cmap,
)
from hypercoil.viz.surfplot import (
    plot_surf_scalars,
    plot_to_image,
)


class TestSurfaceVisualisations:
    @pytest.mark.ci_unsupported
    def test_surf(self):
        surf = CortexTriSurface.from_nmaps(
            template="fsaverage",
            load_mask=True,
            projections=('pial',)
        )
        data = mni152_to_fsaverage(tflow.get(
            template='MNI152NLin2009cAsym',
            suffix='probseg',
            label="GM",
            resolution=2
        ))
        surf.add_gifti_dataset(
            name='gm_density',
            left_gifti=data[0],
            right_gifti=data[1],
            is_masked=False,
            apply_mask=True,
            null_value=None,
        )
        pl, pr = plot_surf_scalars(
            surf=surf,
            projection='pial',
            scalars='gm_density',
        )
        plot_to_image(pl, basename='/tmp/left_density', hemi='left')
        plot_to_image(pr, basename='/tmp/right_density', hemi='right')

        surf = CortexTriSurface.from_tflow(
            load_mask=True,
            projections=('veryinflated', 'inflated', 'sphere')
        )
        parcellation = '/Users/rastkociric/Downloads/desc-schaefer_res-0400_atlas.nii'
        parcellation = '/Users/rastkociric/Downloads/glasser.nii'
        parcellation = '/Users/rastkociric/Downloads/gordon.nii'
        parcellation = pkgrf(
            'hypercoil',
            'viz/resources/nullexample.nii'
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

        (cmap_left, clim_left), (cmap_right, clim_right) = make_cmap(
            surf, 'cmap', 'parcellation')

        # pl = pv.Plotter()
        # surf.left.project('veryinflated')
        # pl.add_mesh(
        #     surf.left,
        #     opacity=1.0,
        #     show_edges=False,
        #     scalars='parcellation',
        #     cmap=cmap_left,
        #     clim=clim_left,
        #     below_color='black',
        # )
        # pl.add_mesh(
        #     surf.left.contour(
        #         isosurfaces=range(int(max(surf.left.point_data['parcellation'])))
        #     ),
        #     color="black",
        #     line_width=5
        # )
        # pl.show(cpos="yz")

        pl, pr = plot_surf_scalars(
            surf,
            projection='veryinflated',
            scalars='parcellation',
            boundary_color='black',
            boundary_width=5,
            cmap=(cmap_left, cmap_right),
            clim=(clim_left, clim_right),
        )
        plot_to_image(pl, basename='/tmp/left', hemi='left')
        plot_to_image(pr, basename='/tmp/right', hemi='right')
        pl, pr = plot_surf_scalars(
            surf,
            projection='inflated',
            scalars='parcellation',
            boundary_color='white',
            boundary_width=2,
            cmap=(cmap_left, cmap_right),
            clim=(clim_left, clim_right),
        )
        plot_to_image(
            pl, basename='/tmp/left', views=((-20, 0, 0),), hemi='left')
        plot_to_image(
            pr, basename='/tmp/right',
            views=(((60, 60, 0), (0, 0, 0), (0, 0, 1)),),
            hemi='right'
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
