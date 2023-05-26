# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for surface plots with None as hemisphere argument
"""
import pytest
import templateflow.api as tflow

from hypercoil.viz.surfplot import plot_surf_scalars
from hypercoil.viz.flows import (
    ichain,
    ochain,
    iochain,
    replicate,
)
from hypercoil.viz.transforms import (
    surf_from_archive,
    resample_to_surface,
    row_major_grid,
    save_fig,
    closest_ortho_cameras,
    scalar_focus_camera,
    planar_sweep_cameras,
    auto_cameras,
)


class TestSurfaceVisualisations:
    @pytest.mark.ci_unsupported
    def test_focused_view_both_hemispheres(self):
        i_chain = ichain(
            surf_from_archive(),
            resample_to_surface(
                'difumo',
                template='fsaverage',
                select=list(range(60)),
                plot=True
            ),
            replicate(map_over=("scalars",)),
            scalar_focus_camera(projection='pial', kind='centroid'),
        )
        o_chain = ochain(
            row_major_grid(ncol=4, figsize=(8, 10), num_panels=20),
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
            difumo_nifti=nii,
            projection='pial',
            filename='/tmp/parcelfocused_index-{index}.png',
            cmap='Purples',
            below_color='white',
            window_size=(400, 250),
        )

    @pytest.mark.ci_unsupported
    def test_ortho_views_both_hemispheres(self):
        i_chain = ichain(
            surf_from_archive(),
            resample_to_surface('difumo', template='fsaverage', plot=True),
            replicate(map_over=("scalars",)),
            closest_ortho_cameras(projection='pial', n_ortho=3),
        )
        o_chain = ochain(
            row_major_grid(ncol=3, figsize=(3, 16), num_panels=48),
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
            difumo_nifti=nii,
            projection='pial',
            filename='/tmp/parcelortho_index-{index}.png',
            cmap='Purples',
            below_color='white',
            window_size=(400, 250),
        )

    @pytest.mark.ci_unsupported
    def test_planar_sweep_both_hemispheres(self):
        i_chain = ichain(
            surf_from_archive(),
            resample_to_surface('difumo', template='fsaverage', plot=True),
            replicate(map_over=("scalars",)),
            planar_sweep_cameras(initial=(1, 0, 0), n_steps=5),
        )
        o_chain = ochain(
            row_major_grid(ncol=10, figsize=(10, 10), num_panels=100),
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
            difumo_nifti=nii,
            projection='pial',
            filename='/tmp/parcelplanar_index-{index}.png',
            cmap='Purples',
            below_color='white',
            window_size=(400, 250),
        )

    @pytest.mark.ci_unsupported
    def test_auto_view_both_hemispheres(self):
        i_chain = ichain(
            surf_from_archive(),
            resample_to_surface('difumo', template='fsaverage', plot=True),
            replicate(map_over=("scalars",)),
            auto_cameras(projection='pial', n_ortho=3, focus='peak', n_angles=3),
        )
        o_chain = ochain(
            row_major_grid(ncol=14, figsize=(14, 6), num_panels=84),
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
            difumo_nifti=nii,
            projection='pial',
            filename='/tmp/parcelauto_index-{index}.png',
            cmap='Purples',
            below_color='white',
            window_size=(400, 250),
        )
