# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for surface plots with specified hemisphere arguments
"""
import pytest

from pkg_resources import resource_filename as pkgrf

import jax
import templateflow.api as tflow

from hypercoil.init.atlas import DirichletInitSurfaceAtlas
from hypercoil.viz.surfplot import plot_surf_scalars
from hypercoil.viz.flows import (
    ichain,
    ochain,
    iochain,
    replicate,
)
from hypercoil.viz.transforms import (
    surf_from_archive,
    scalars_from_atlas,
    row_major_grid,
    save_fig,
    closest_ortho_cameras,
    scalar_focus_camera,
    planar_sweep_cameras,
    auto_cameras,
)


class TestSurfaceVisualisationsSingleHemisphere:
    @pytest.mark.ci_unsupported
    def test_focused_view_single_hemisphere(self):
        i_chain = ichain(
            surf_from_archive(),
            scalars_from_atlas('dirichlet', plot=True),
            replicate(map_over=("scalars", "hemi")),
            scalar_focus_camera(projection='veryinflated', kind='peak'),
        )
        o_chain = ochain(
            row_major_grid(ncol=4, figsize=(8, 10), num_panels=20),
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
            dirichlet_atlas=atlas,
            projection='veryinflated',
            filename='/tmp/dirfocused_index-{index}.png',
        )

    @pytest.mark.ci_unsupported
    def test_ortho_views_single_hemisphere(self):
        selected = list(range(19)) + list(range(20, 39))
        i_chain = ichain(
            surf_from_archive(),
            scalars_from_atlas('dirichlet', select=selected, plot=True),
            replicate(map_over=("scalars", "hemi")),
            closest_ortho_cameras(projection='veryinflated', n_ortho=3),
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
            dirichlet_atlas=atlas,
            projection='veryinflated',
            filename='/tmp/dirortho_index-{index}.png',
        )

    @pytest.mark.ci_unsupported
    def test_planar_sweep_single_hemisphere(self):
        i_chain = ichain(
            surf_from_archive(),
            scalars_from_atlas('dirichlet', plot=True),
            replicate(map_over=("scalars", "hemi")),
            planar_sweep_cameras(initial=(1, 0, 0), normal=(0, 0, 1), n_steps=10),
        )
        o_chain = ochain(
            row_major_grid(ncol=10, figsize=(10, 10), num_panels=100),
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
            dirichlet_atlas=atlas,
            projection='veryinflated',
            filename='/tmp/dirplanar_index-{index}.png',
        )

    @pytest.mark.ci_unsupported
    def test_auto_view_single_hemisphere(self):
        i_chain = ichain(
            surf_from_archive(),
            scalars_from_atlas('dirichlet', plot=True),
            replicate(map_over=("scalars", "hemi")),
            auto_cameras(projection='veryinflated', n_ortho=3, focus='peak', n_angles=3),
        )
        o_chain = ochain(
            row_major_grid(ncol=7, figsize=(7, 10), num_panels=70),
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
            dirichlet_atlas=atlas,
            projection='veryinflated',
            filename='/tmp/dirauto_index-{index}.png',
        )
