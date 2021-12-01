# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Canonical workflow
~~~~~~~~~~~~~~~~~~
Standard functional connectivity workflow. Nothing differentiable about it.
"""
from hypercoil.neuro.atlas import fsLRAtlas
from hypercoil.init import (
    IIRFilterSpec
)
from hypercoil.functional import corr
from hypercoil.nn import (
    AtlasLinear,
    FrequencyDomainFilter,
    UnaryCovarianceUW
)
from hypercoil.data import HCPDataset
from hypercoil.data.dataset import ReferencedDataLoader


# Temporary hard-codes
data_dir = '/mnt/andromeda/Data/HCP_subsubsample/'
atlas_path = '/home/rastko/Downloads/atlases/glasser.nii'


# Small sample
ds = HCPDataset(
    data_dir,
    model=['(dd1(rps + wm + csf + gsr))^^2']
)
ds.set_depth(1)
dl = ReferencedDataLoader(ds, batch_size=3, shuffle=True)
for sample in dl:
    break

time_dim = sample['images'].shape[-1]


glasser = fsLRAtlas(path=atlas_path, name='glasser')
atlas = AtlasLinear(glasser, mask_input=False)

bp_spec = IIRFilterSpec(Wn=(0.01, 0.08), N=1, ftype='butter', fs=(1 / 0.72))
bp = FrequencyDomainFilter(
    filter_specs=[bp_spec],
    time_dim=time_dim
)

cov = UnaryCovarianceUW(
    dim=time_dim,
    estimator=corr
)

X = sample['images']
X = atlas(X)
X = torch.cat([X['cortex_L'], X['cortex_R']], -2)
X = bp(X).squeeze(1)
X = cov(X)
