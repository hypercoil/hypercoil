# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for volplot-based visualisations
"""
import pytest
import nibabel as nb
import numpy as np


from hypercoil.viz.surf import (
    CortexTriSurface,
)
from hypercoil.viz.volplot import (
    plot_embedded_volume,
)


class TestNetworkVisualisations:
    @pytest.mark.ci_unsupported
    def test_vol(self):
        surf = CortexTriSurface.from_nmaps(
            template="fsaverage",
            load_mask=True,
            projections=('pial',)
        )
        nii = nb.load("/Users/rastkociric/Downloads/pain_thresh_cFWE05.nii.gz")
        plot_embedded_volume(
            surf=surf,
            nii=nii,
            point_size=5,
            cmap='magma',
        )
