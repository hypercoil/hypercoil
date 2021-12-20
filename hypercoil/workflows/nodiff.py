# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Canonical workflow
~~~~~~~~~~~~~~~~~~
Standard functional connectivity workflow. Nothing differentiable about it.
"""
import torch
from functools import partial
from hypercoil.neuro.atlas import fsLRAtlas
from hypercoil.init import (
    IIRFilterSpec
)
from hypercoil.functional import corr, conditionalcorr
from hypercoil.nn import (
    AtlasLinear,
    FrequencyDomainFilter,
    UnaryCovarianceUW,
    BinaryCovarianceUW,
    Residualise
)
from hypercoil.data import HCPDataset
from hypercoil.data.dataset import ReferencedDataLoader
from hypercoil.data.functional import window_map, identity
from hypercoil.data.transforms import (
    Compose,
    PolynomialDetrend,
    Normalise,
    NaNFill
)
from hypercoil.data.wds import torch_wds


# Temporary hard-codes
data_dir = '/mnt/andromeda/Data/HCP_wds/spl-{000..003}_shd-{000000..000003}.tar'
atlas_path = '/home/rastko/Downloads/atlases/glasser.nii'
batch_size = 10
buffer_size = 20


# Small sample
wndw = partial(
    window_map,
    keys=('images', 'confounds', 'tmask'),
    window_length=500
)
dmdt = Compose([NaNFill(0), PolynomialDetrend(1), Normalise()])
maps = {
    'images' : dmdt,
    'confounds' : dmdt,
    'tmask' : identity,
    't_r' : identity,
    'task' : identity,
}
ds = torch_wds(
    data_dir,
    keys=maps,
    batch_size=batch_size,
    shuffle=buffer_size,
    map=wndw
)
for sample in ds:
    rest = sample[-1][:, -3]
    if rest.sum() >= 3:
        break
sample = tuple([s[rest.bool()] for s in sample])
batch_size = sample[0].shape[0]

time_dim = sample[0].shape[-1]


glasser = fsLRAtlas(path=atlas_path, name='glasser')
atlas = AtlasLinear(glasser, mask_input=False)

bp_spec = IIRFilterSpec(Wn=(0.01, 0.08), ftype='ideal', fs=(1 / 0.72))
bp = FrequencyDomainFilter(
    filter_specs=[bp_spec],
    time_dim=time_dim
)

res = Residualise()
cov = UnaryCovarianceUW(
    dim=time_dim,
    estimator=corr
)
ccov = BinaryCovarianceUW(
    dim=time_dim,
    estimator=conditionalcorr
)

X = sample[0]
K = sample[1]
M = sample[2].unsqueeze(-2)

X = atlas(X)
X = torch.cat([X['cortex_R'], X['cortex_L']], -2)
X = bp(X).squeeze(1)
K = bp(K).squeeze(1)
C = ccov(X, K, mask=M)
X = res(X, K, mask=M)
C2 = cov(X, mask=M)


# CANONICAL PIPELINE FOR REFERENCE
import numpy as np
from scipy.fft import fft, ifft, fftfreq

atlasR = glasser.ref.get_fdata().squeeze()
#bpR = butter(N=1, Wn=(0.01, 0.08), btype='bandpass', fs=(1 / 0.72))

XR = sample[0].numpy()[:, :atlasR.shape[-1]]
KR = sample[1].numpy()
MR = sample[2].numpy().astype(bool)

labels = np.unique(atlasR)
label_mean_ts = np.empty((batch_size, len(labels), time_dim))
for i, label in enumerate(labels):
    label_mean_ts[:, i] = XR[:, atlasR==label].mean(-2)
XR = label_mean_ts

def bpR(data, lpf, hpf):
    freq = fftfreq(time_dim, d=0.72)
    dataf = fft(data)
    mask = np.logical_and(np.abs(freq) > hpf, np.abs(freq) < lpf)
    dataf[..., ~mask] = 0
    return ifft(dataf).real

XR = bpR(XR, 0.08, 0.01)
KR = bpR(KR, 0.08, 0.01)
#XR = filtfilt(bpR[0], bpR[1], XR, axis=-1)
#KR = filtfilt(bpR[0], bpR[1], KR, axis=-1)

CR = np.empty((batch_size, XR.shape[1], XR.shape[1]))
for i, (x, k, m) in enumerate(zip(XR, KR, MR)):
    x = x[:, m]
    k = k[:, m]
    betas, _, _, _ = np.linalg.lstsq(k.T, x.T)
    x = x - betas.T @ k
    CR[i] = np.corrcoef(x)
