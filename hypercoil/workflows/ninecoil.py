# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Ninefold workflow
~~~~~~~~~~~~~~~~~
A standard functional connectivity workflow with frozen parameters, coupled
with two learnable atlases and two learnable filters, yielding 9 connectivity
channels to the model. For the model we use a residual sylo network.
"""
from hypercoil.neuro.atlas import fsLRAtlas
from hypercoil.init import (
    IIRFilterSpec
)
from hypercoil.functional import corr, conditionalcorr
from hypercoil.nn import (
    AtlasLinear,
    FrequencyDomainFilter,
    BinaryCovarianceUW,
)


# Standard architecture
glasser = fsLRAtlas(path=atlas_path, name='glasser')
atlas = AtlasLinear(glasser, mask_input=False)
bp_spec = IIRFilterSpec(Wn=(0.01, 0.1), N=1, ftype='butter', fs=(1 / 0.72))
bp = FrequencyDomainFilter(
    filter_specs=[bp_spec],
    time_dim=time_dim
)
ccov = BinaryCovarianceUW(
    dim=time_dim,
    estimator=conditionalcorr
)

# Learnable atlases
atlasL = [
    AtlasLinear(glasser, mask_input=False, kernel_sigma=1.5, noise_sigma=0.02),
    AtlasLinear(glasser, mask_input=False, kernel_sigma=4, noise_sigma=0.02)
]
# Learnable filters
filterL = [
    FrequencyDomainFilter()
]
