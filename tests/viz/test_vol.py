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
    ichain,
    ochain,
    iochain,
    split_chain,
    map_over_sequence,
)
from hypercoil.viz.transforms import (
    vol_from_nifti,
    vol_from_atlas,
    row_major_grid,
    col_major_grid,
    save_fig,
    plot_and_save,
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
        i_chain = vol_from_nifti()
        o_chain = plot_and_save()
        f = iochain(plot_embedded_volume, i_chain, o_chain)
        out = f(
            surf=surf,
            nii=nii,
            cmap='magma',
            basename='/tmp/vol',
            views=("dorsal", "left", "anterior"),
            hemi='both',
        )
        assert len(out.keys()) == 1
        assert "screenshots" in out.keys()

        atlas = MultifileVolumetricAtlas(
            ref_pointer=[tflow.get(
                template='MNI152NLin2009cAsym',
                suffix='probseg',
                label=l,
                resolution=2)
            for l in ('CSF', 'GM', 'WM')],
            clear_cache=False
        )
        i_chain = ichain(
            vol_from_atlas(),
            apply_along_axis(var="val", axis=0),
        )
        o_chain = ochain(
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
        f = iochain(plot_embedded_volume, i_chain, None)
        f = iochain(plot_embedded_volume, i_chain, o_chain)
        out = f(
            surf=surf,
            atlas=atlas,
            cmap='magma',
            views=("dorsal", "left", "anterior"),
            hemi="both",
        )
