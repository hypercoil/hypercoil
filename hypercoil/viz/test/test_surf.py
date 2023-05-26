# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for elementary surfplot-based visualisations
"""
import pytest

from pkg_resources import resource_filename as pkgrf

import numpy as np
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
    parcellate_scalars,
    parcellate_colormap,
    scatter_into_parcels,
    save_html,
)


class TestSurfaceVisualisations:
    @pytest.mark.ci_unsupported
    def test_scalars(self):
        i_chain = ichain(
            surf_from_archive(),
            resample_to_surface('gm_density', template='fsaverage', plot=True),
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
            gm_density_nifti=tflow.get(
                template='MNI152NLin2009cAsym',
                suffix='probseg',
                label="GM",
                resolution=2
            ),
            projection='pial',
        )
        assert len(out.keys()) == 1
        assert "screenshots" in out.keys()

    @pytest.mark.ci_unsupported
    def test_parcellation(self):
        i_chain = ichain(
            surf_from_archive(),
            scalars_from_cifti('parcellation', plot=True),
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
            parcellation_cifti=pkgrf(
                'hypercoil',
                'viz/resources/nullexample.nii'
            ),
            projection='veryinflated',
            boundary_color='black',
            boundary_width=5,
        )

    @pytest.mark.ci_unsupported
    def test_parcellation_modal_cmap(self):
        i_chain = ichain(
            surf_from_archive(),
            scalars_from_cifti('parcellation', plot=True),
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
            parcellation_cifti=pkgrf(
                'hypercoil',
                'viz/resources/nullexample.nii'
            ),
            projection='veryinflated',
            boundary_color='black',
            boundary_width=5,
        )

    @pytest.mark.ci_unsupported
    def test_parcellation_html(self):
        i_chain = ichain(
            surf_from_archive(),
            scalars_from_cifti('parcellation', plot=True),
            parcellate_colormap('network', 'parcellation')
        )
        o_chain = ochain(
            map_over_sequence(
                xfm=save_html(backend="panel"),
                mapping={
                    "filename": ('/tmp/left.html', '/tmp/right.html'),
                }
            ),
        )
        f = iochain(plot_surf_scalars, i_chain, o_chain)
        f(
            template="fsLR",
            load_mask=True,
            parcellation_cifti=pkgrf(
                'hypercoil',
                'viz/resources/nullexample.nii'
            ),
            projection='veryinflated',
            boundary_color='black',
            boundary_width=5,
        )

    @pytest.mark.ci_unsupported
    def test_parcellated_scalars(self):
        i_chain = ichain(
            surf_from_archive(),
            resample_to_surface('gm_density', template='fsLR', plot=True),
            scalars_from_cifti('parcellation'),
            parcellate_scalars('gm_density', 'parcellation'),
        )
        o_chain = ochain(
            map_over_sequence(
                xfm=plot_and_save(),
                mapping={
                    "basename": ('/tmp/left_density_parc', '/tmp/right_density_parc'),
                    "hemi": ('left', 'right'),
                }
            )
        )
        f = iochain(plot_surf_scalars, i_chain, o_chain)
        out = f(
            template="fsLR",
            load_mask=True,
            parcellation_cifti=pkgrf(
                'hypercoil',
                'viz/resources/nullexample.nii'
            ),
            gm_density_nifti=tflow.get(
                template='MNI152NLin2009cAsym',
                suffix='probseg',
                label="GM",
                resolution=2
            ),
            projection='inflated',
            clim=(0.2, 0.9),
        )
        assert len(out.keys()) == 1
        assert "screenshots" in out.keys()

        parcellated = np.random.rand(400)
        i_chain = ichain(
            surf_from_archive(),
            scalars_from_cifti('parcellation'),
            scatter_into_parcels('scalars', 'parcellation'),
        )
        o_chain = ochain(
            map_over_sequence(
                xfm=plot_and_save(),
                mapping={
                    "basename": ('/tmp/left_noise_parc', '/tmp/right_noise_parc'),
                    "hemi": ('left', 'right'),
                }
            )
        )
        f = iochain(plot_surf_scalars, i_chain, o_chain)
        out = f(
            template="fsLR",
            load_mask=True,
            parcellation_cifti=pkgrf(
                'hypercoil',
                'viz/resources/nullexample.nii'
            ),
            parcellated=parcellated,
            projection='inflated',
            clim=(0, 1),
            cmap='inferno',
        )
