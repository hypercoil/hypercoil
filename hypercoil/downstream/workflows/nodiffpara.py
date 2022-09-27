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
from hypercoil.init.iirfilterbutforreal import (
    IIRFilterSpec
)
from hypercoil.functional import corr, conditionalcorr
from hypercoil.nn import (
    AtlasLinear,
    DTDF,
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
gordon_path = '/home/rastko/Downloads/atlases/gordon.nii'
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
gordon = fsLRAtlas(path=gordon_path, name='gordon')
atlas = AtlasLinear(glasser, mask_input=False)
atlas2 = AtlasLinear(glasser, mask_input=False)
for p in atlas2.parameters(): p.requires_grad = False
gdn = gordon.map()
atlas2.weight['cortex_L'][:] = 0
atlas2.weight['cortex_R'][:] = 0
atlas2.weight['cortex_L'][:161] = gdn['cortex_L']
atlas2.weight['cortex_R'][:172] = gdn['cortex_R']
for p in atlas2.parameters(): p.requires_grad = True

atlas = {
    'glasser': atlas,
    'gordon': atlas2
}


conf_36 = torch.eye(36)
conf_24 = torch.eye(36)
conf_24[6:9] = 0
conf_24[15:18] = 0
conf_24[24:27] = 0
conf_24[33:] = 0
conf_24 = conf_24[conf_24.sum(-1) == 1]
conf_9 = torch.eye(36)
conf_9[9:] = 0
conf_9 = conf_9[conf_9.sum(-1) == 1]
conf = {
    '36p': conf_36,
    '24p': conf_24,
    '9p': conf_9
}


bp_spec = {
    'bwbp': IIRFilterSpec(Wn=(0.01, 0.08), ftype='butter', fs=(1 / 0.72)),
    'bw2bb' : IIRFilterSpec(Wn=(0.01, 0.15), ftype='butter', N=2, fs=(1 / 0.72)),
    'ch2hp' : IIRFilterSpec(Wn=0.01, btype='highpass', ftype='cheby1', N=2, fs=(1 / 0.72))
}
bp = {n :DTDF(spec=spec,) for n, spec in bp_spec.items()}

res = Residualise()
cov = UnaryCovarianceUW(
    dim=time_dim,
    estimator=corr
)
ccov = BinaryCovarianceUW(
    dim=time_dim,
    estimator=conditionalcorr
)

C, C2 = {}, {}
for a_name, a in atlas.items():
    for m_name, m in conf.items():
        for f_name, f in bp.items():
            X = sample[0]
            K = sample[1]
            M = sample[2].unsqueeze(-2)

            K = m @ K
            X = a(X)
            X = torch.cat([X['cortex_R'], X['cortex_L']], -2)
            idx = ~X[0, :, 0].isnan()
            idx2 = (idx.float().view(-1, 1) @ idx.float().view(1, -1)).bool()
            X = X[:, idx, :]
            X = f(X).squeeze(1)
            K = f(K).squeeze(1)
            conn = torch.empty((batch_size, 360, 360)) * torch.nan
            conn[idx2.tile(batch_size, 1, 1)] = ccov(X, K, mask=M).ravel()
            C[f'{a_name}_{m_name}_{f_name}'] = conn
            X = res(X, K, mask=M)
            conn = torch.empty((batch_size, 360, 360)) * torch.nan
            conn[idx2.tile(batch_size, 1, 1)] = cov(X, mask=M).ravel()
            C2[f'{a_name}_{m_name}_{f_name}'] = conn


# CANONICAL PIPELINE FOR REFERENCE
import numpy as np
#from scipy.fft import fft, ifft, fftfreq
from scipy.signal import butter, lfilter

atlasR = glasser.ref.get_fdata().squeeze()
bpR = butter(N=1, Wn=(0.01, 0.08), btype='bandpass', fs=(1 / 0.72))

XR = sample[0].numpy()[:, :atlasR.shape[-1]]
KR = sample[1].numpy()
MR = sample[2].numpy().astype(bool)

labels = np.unique(atlasR)
label_mean_ts = np.empty((batch_size, len(labels), time_dim))
for i, label in enumerate(labels):
    label_mean_ts[:, i] = XR[:, atlasR==label].mean(-2)
XR = label_mean_ts

"""
def bpR(data, lpf, hpf):
    freq = fftfreq(time_dim, d=0.72)
    dataf = fft(data)
    mask = np.logical_and(np.abs(freq) > hpf, np.abs(freq) < lpf)
    dataf[..., ~mask] = 0
    return ifft(dataf).real
"""

#XR = bpR(XR, 0.08, 0.01)
#KR = bpR(KR, 0.08, 0.01)
XR = lfilter(bpR[0], bpR[1], XR, axis=-1)
KR = lfilter(bpR[0], bpR[1], KR, axis=-1)

CR = np.empty((batch_size, XR.shape[1], XR.shape[1]))
for i, (x, k, m) in enumerate(zip(XR, KR, MR)):
    x = x[:, m]
    k = k[:, m]
    betas, _, _, _ = np.linalg.lstsq(k.T, x.T)
    x = x - betas.T @ k
    CR[i] = np.corrcoef(x)
