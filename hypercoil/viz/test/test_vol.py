# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for volplot-based visualisations
"""
import pytest
import nibabel as nb
import numpy as np
import templateflow.api as tflow

from hypercoil.init.atlas import MultifileVolumetricAtlas
from hypercoil.viz.surf import (
    CortexTriSurface,
)
from hypercoil.viz.utils import (
    plot_to_image,
    plot_to_display,
)
from hypercoil.viz.volplot import (
    plot_embedded_volume,
)
from hypercoil.viz.flows import (
    apply_along_axis,
    source_chain,
    sink_chain,
    transform_chain,
    split_chain,
    map_over_sequence,
)
from hypercoil.viz.transforms import (
    vol_from_nifti,
    vol_from_atlas,
    row_major_grid,
    col_major_grid,
    save_fig,
)


class TestVolumeVisualisations:
    @pytest.mark.ci_unsupported
    def test_vol(self):
        surf = CortexTriSurface.from_nmaps(
            template="fsaverage",
            load_mask=True,
            projections=('pial',)
        )
        nii = nb.load("/Users/rastkociric/Downloads/pain_thresh_cFWE05.nii.gz")
        f = vol_from_nifti()(plot_embedded_volume)
        p = f(
            surf=surf,
            nii=nii,
            #point_size=5,
            cmap='magma',
            #off_screen=False,
        )
        #plot_to_display(p)
        plot_to_image(
            p,
            basename='/tmp/vol',
            views=("dorsal", "left", "anterior"),
            hemi='both'
        )

        atlas = MultifileVolumetricAtlas(
            ref_pointer=[tflow.get(
                template='MNI152NLin2009cAsym',
                suffix='probseg',
                label=l,
                resolution=2)
            for l in ('CSF', 'GM', 'WM')],
            clear_cache=False
        )
        src_chain = source_chain(
            vol_from_atlas(),
            apply_along_axis(var="val", axis=0),
        )
        snk_chain = sink_chain(
            split_chain(
                row_major_grid(nrow=3, figsize=(30, 20)),
                col_major_grid(ncol=3, figsize=(30, 20)),
            ),
            map_over_sequence(
                xfm=save_fig(),
                mapping={
                    "filename": (
                        "/tmp/probseg_row.png",
                        "/tmp/probseg_col.png",
                    )
                },
            ),
        )
        f = transform_chain(plot_embedded_volume, src_chain, snk_chain)
        f(
            surf=surf,
            atlas=atlas,
            cmap='magma',
            views=("dorsal", "left", "anterior"),
            hemi="both",
        )
