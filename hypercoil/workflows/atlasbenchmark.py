# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlas benchmarks
~~~~~~~~~~~~~~~~
Benchmark parcellations.
"""
import torch
import pathlib
import numpy as np
import nibabel as nb
import templateflow.api as tflow
from collections import OrderedDict
from hypercoil.eval.atlas import AtlasEval
from hypercoil.functional import (
    corr,
    residualise
)
from hypercoil.data.functional import identity
from hypercoil.data.transforms import (
    Normalise
)
from hypercoil.data.wds import torch_wds


atlas_root = '/mnt/pulsar/Data/atlases/spatialnulls'
data_dir = '/mnt/andromeda/Data/HCP_wds/spl-008_shd-{000000..000004}.tar'
batch_size = 1
buffer_size = 1
atlas_dim = 59412
device = 'cuda:1'


normalise = Normalise()
maps = {
    'images' : identity,
    'confounds' : identity,
    'tmask' : identity,
    't_r' : identity,
    'task' : identity,
}
ds = torch_wds(
    data_dir,
    keys=maps,
    batch_size=batch_size,
    shuffle=buffer_size,
    map=identity
)
polybasis = torch.stack([
    torch.arange(1200, device=device) ** i
    for i in range(3)
])

mask = torch.zeros(91282, dtype=torch.bool, device=device)
mask[:atlas_dim] = 1
eval = AtlasEval(
    mask=mask
)

print('[ Loading parcellations ]')
parcellations = list(pathlib.Path(atlas_root).glob(
    'parcellation*/parcellation*.nii'))
parcellations.sort()
for i, atlas in enumerate(parcellations):
    eval.add_voxel_assignment(
        name=f'labels-200_iter-{i:06}_null',
        asgt=torch.tensor(
            nb.load(atlas).get_fdata(),
            dtype=torch.long,
            device=device
        )
    )
    break

for s, sample in enumerate(ds):
    print(f'[ Preparing next sample {s} ]')
    X = sample[0].squeeze().to(device=device, dtype=torch.float)
    if X.shape[-1] != 1200:
        print(f'[ Incorrect time dim: {X.shape[-1]}. Skipping ]')
        continue
    gs = X.mean(-2, keepdim=True)
    regs = torch.cat((polybasis, gs), -2)
    data = residualise(X, regs, driver='gels')
    data = normalise(data)

    break
