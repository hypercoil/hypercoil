#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Covariance: time-by-time
~~~~~~~~~~~~~~~~~~~~~~~~
Time by time convariance experiment. Hard codes everywhere until we have time
to clean it up.
"""
import os
import torch
import pathlib
import pandas as pd
import webdataset as wds
import templateflow.api as tflow
from functools import partial
from pkg_resources import resource_filename as pkgrf
from hypercoil.data.transforms import Normalise
from hypercoil.data.collate import gen_collate, extend_and_bind
from hypercoil.engine import (
    Epochs, ModelArgument, UnpackingModelArgument, SWA, SWAPR
)
from hypercoil.functional import corr, sym2vec
from hypercoil.functional.domainbase import Identity
from hypercoil.functional.domain import MultiLogit
from hypercoil.functional.matrix import toeplitz
from hypercoil.functional.resid import residualise
from hypercoil.init.atlas import (
    CortexSubcortexCIfTIAtlas,
    DirichletInitSurfaceAtlas
)
from hypercoil.init.dirichlet import DirichletInit
from hypercoil.init.freqfilter import FreqFilterSpec
from hypercoil.loss import (
    LossScheme,
    LossApply,
    Entropy,
    Equilibrium,
    SymmetricBimodal,
    VectorDispersion
)
from hypercoil.loss.jsdiv import JSDivergence
from hypercoil.nn.atlas import AtlasLinear
from hypercoil.nn.cov import UnaryCovariance
from hypercoil.nn.interpolate import HybridInterpolate
from hypercoil.nn.freqfilter import FrequencyDomainFilter
from hypercoil.nn.window import WindowAmplifier
from hypercoil.viz.surf import fsLRAtlasParcels


n_ts = 400
fs = 0.72
lr = 0.005
lr_swa = 0.5
wd = 1e-1
momentum=0.9
batch_size = 20
batch_size_val = 200
window_size = 800
device = 'cuda:0'
save_dev = 'cpu'
total_n_regs = 57
augmentation_factor = 1
n_channels = 7
log_interval = 10
entropy_nu = 0.00001
equilibrium_nu = 1000
dispersion_nu = 0.001
symbimodal_nu = 50
jsdiv_nu = 5000
max_epoch = 1001
discount = 10
resume_epoch = -1
instances_per_epoch = 300
swa = True
atlas = 'schaefer'
dtype = torch.float
confounds_path = '/tmp/confounds/sub-{}_ses-{}_run-{}.pt'
cols = ['framewise_displacement', 'std_dvars'] # we probably won't actually need these here
confounds_path_orig = (
    '/mnt/andromeda/Data/HCP_S1200/data/{}/MNINonLinear/'
    'Results/rfMRI_REST{}_{}/Confound_Regressors.tsv')
results = f'/tmp/timecov{n_channels:03}'

if atlas == 'schaefer':
    train_dir = '/mnt/andromeda/Data/HCP_parcels_wds/spl-{000..005}_shd-{000000..000004}.tar'
    val_dir = '/mnt/andromeda/Data/HCP_parcels_wds/spl-{006..007}_shd-{000000..000004}.tar'
    ref_pointer = '/mnt/pulsar/Data/atlases/atlases/desc-schaefer_res-0400_atlas.nii'
elif atlas == 'SWAPR':
    train_dir = '/mnt/andromeda/Data/HCP_SWAPRparcels_wds/spl-{000..005}_shd-{000000..000004}.tar'
    val_dir = '/mnt/andromeda/Data/HCP_SWAPRparcels_wds/spl-{006..007}_shd-{000000..000004}.tar'
    ref_pointer = '/mnt/pulsar/Data/atlases/atlases/desc-SWAPR_res-000200_atlas.nii'

atlas_ref = CortexSubcortexCIfTIAtlas(
    ref_pointer=ref_pointer,
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
    clear_cache=False,
    dtype=dtype,
    device=device
)
atlas_ref = AtlasLinear(atlas_ref, device=device)
atlas_plot = DirichletInitSurfaceAtlas(
    cifti_template=ref_pointer,
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
        'cortex_L': n_channels,
        'cortex_R': n_channels,
        'subcortex': 0
    },
    conc=100.,
    dtype=dtype,
    device=device
)
atlas_plot = AtlasLinear(atlas_plot, device=device)


def data_cfg(train_dir, val_dir, batch_size, batch_size_val):
    collate = partial(gen_collate, concat=extend_and_bind)
    dpl = wds.DataPipeline(
        wds.ResampledShards(train_dir),
        wds.shuffle(100),
        wds.tarfile_to_samples(1),
        wds.decode(lambda x, y: wds.torch_loads(y)),
        wds.shuffle(batch_size),
    )
    dl = torch.utils.data.DataLoader(
        dpl, num_workers=2, batch_size=batch_size, collate_fn=collate)

    dpl_val = wds.DataPipeline(
        wds.ResampledShards(val_dir),
        wds.shuffle(100),
        wds.tarfile_to_samples(1),
        wds.decode(lambda x, y: wds.torch_loads(y)),
        wds.shuffle(batch_size_val),
    )
    dl_val = torch.utils.data.DataLoader(
        dpl_val, num_workers=2, batch_size=batch_size_val, collate_fn=collate)
    return dl, dl_val


def model_cfg(window_size, device):
    interpol = HybridInterpolate()
    fft_init = FreqFilterSpec(
        Wn=(0.01, 0.1),
        ftype='ideal',
        btype='bandpass',
        fs=fs
    )
    bpfft = FrequencyDomainFilter(
        filter_specs=[fft_init],
        time_dim=window_size,
        domain=Identity(),
        device=device
    )
    bpfft.preweight.requires_grad = False
    selector = torch.zeros((36, total_n_regs), device=device)
    selector[:36, :36] = torch.eye(36, device=device)
    return interpol, bpfft, selector


def loss_cfg(max_epoch, resume_epoch):
    epochs = Epochs(max_epoch)
    loss = LossScheme([
        LossScheme([
            Entropy(nu=entropy_nu, axis=0),
            Equilibrium(nu=equilibrium_nu)
        ], apply=lambda arg: arg.weight),
        LossScheme([
            VectorDispersion(nu=dispersion_nu),
            SymmetricBimodal(nu=symbimodal_nu, modes=(-1, 1))
        ], apply=lambda arg: sym2vec(arg.corr)),
        #LossApply(
        #    JSDivergence(nu=jsdiv_nu),
        #    apply=lambda arg: UnpackingModelArgument(P=arg.lh, Q=arg.rh)
        #)
    ])
    return epochs, loss


def get_confs(keys):
    basenames = [d.split('/')[-1] for d in keys]
    subs = [b.split('_')[0].split('-')[-1] for b in basenames]
    sess = [b.split('_')[2].split('-')[-1] for b in basenames]
    runs = [b.split('_')[3].split('-')[-1] for b in basenames]
    confs = [(confounds_path.format(sub, ses, run), (sub, ses, run))
             for sub, ses, run in zip(subs, sess, runs)]
    confs_dne = [(confounds_path_orig.format(*attr), attr)
                 for path, attr in confs
                 if not pathlib.Path(path).is_file()]
    confs_dne_out = [confounds_path.format(*attr) for (_, attr) in confs_dne]
    confs_dne = [pd.read_csv(path, sep='\t') for path, attr in confs_dne]
    for c_df, path in zip(confs_dne, confs_dne_out):
        torch.save(
            torch.tensor(c_df[cols].values.T, dtype=dtype, device=save_dev),
            path)
    confs = extend_and_bind([torch.load(c[0]) for c in confs])
    return confs


def ingest_data(dl, wndw):
    n = Normalise()
    data = next(dl)
    qc = get_confs(data['__key__'])[:, 0]
    bold = n(data['images'])
    confounds = n(data['confounds'])
    nanmask = (torch.isnan(bold).sum(-2) == 0)
    bold[torch.isnan(bold)] = 0.001 * torch.randn_like(bold[torch.isnan(bold)])
    confounds[torch.isnan(confounds)] = 0.001 * torch.randn_like(
        confounds[torch.isnan(confounds)])
    qc[torch.isnan(qc)] = 0.001 * torch.randn_like(qc[torch.isnan(qc)])

    [bold, confounds, qc, nanmask] = wndw((bold, confounds, qc, nanmask))
    return (
        bold.to(device=device),
        confounds.to(device=device),
        qc.to(device=device),
        nanmask.to(device=device)
    )


def preprocess(bold, confs, mask, interpol, bpfft, selector):
    with torch.no_grad():
        assert mask.dtype == torch.bool
        confs = selector @ confs
        bold = interpol(bold.unsqueeze(1), mask).squeeze()
        bold = bpfft(bold).squeeze()
        confs = interpol(confs.unsqueeze(1), mask).squeeze()
        confs = bpfft(confs).squeeze()
        return residualise(bold, confs, driver='gels')


modal_cmap = pkgrf(
    'hypercoil',
    'viz/resources/cmap_modal.nii'
)
network_cmap = pkgrf(
    'hypercoil',
    'viz/resources/cmap_network.nii'
)
os.makedirs(results, exist_ok=True)

dl, dl_val = data_cfg(
    train_dir=train_dir,
    val_dir=val_dir,
    batch_size=batch_size,
    batch_size_val=batch_size_val
)
interpol, bpfft, selector = model_cfg(
    window_size=window_size,
    device=device
)

cov_init = DirichletInit(
    n_classes=n_channels,
    axis=0,
    concentration=torch.tensor([100.0 for _ in range(n_channels)])
)
cov = UnaryCovariance(
    dim=n_ts,
    estimator=corr,
    out_channels=n_channels,
    init=cov_init,
    device=device
)
opt = torch.optim.SGD(
    lr=lr, momentum=momentum, weight_decay=wd,
    params=cov.parameters())
epochs, loss = loss_cfg(max_epoch, resume_epoch)
start_swa = float('inf')
batches_per_epoch = instances_per_epoch // batch_size
if swa:
    cov_swa = torch.optim.swa_utils.AveragedModel(cov)
    scheduler_lr = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=opt,
        milestones=(100, 150, 200),
        gamma=0.5,
        verbose=True)
    scheduler_swa = torch.optim.swa_utils.SWALR(
        optimizer=opt,
        swa_lr=lr_swa)
    start_swa = 200
    #swa = SWAPR(
    swa = SWA(
        epochs=epochs,
        swa_start=start_swa,
        swa_model=cov_swa,
        swa_scheduler=scheduler_swa,
        model=cov,
        scheduler=scheduler_lr,
        #revolve_epochs=tuple(range(start_swa + 50, max_epoch - 1, 50)),
        #device=device
    )

wndw = WindowAmplifier(
    window_size=window_size,
    augmentation_factor=augmentation_factor
)

discount_matrix = 1 - toeplitz(1 / (torch.arange(window_size, device=device) / discount).exp())

for epoch in epochs:
    iter_dl = iter(dl)
    print(f'\n[ Epoch {epoch} ]')
    for _ in range(batches_per_epoch):
        bold, confounds, qc, nanmask = ingest_data(iter_dl, wndw)

        denoised = preprocess(bold, confounds, nanmask, interpol, bpfft, selector)
        time_by_time = cov(denoised.transpose(-2, -1)) * discount_matrix
        #print(torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated())
        #continue
        #print(denoised)
        loss_arg = ModelArgument(
            weight=cov.weight,
            corr=time_by_time,
            lh=cov.weight.squeeze().t()[:(n_ts // 2)],
            rh=cov.weight.squeeze().t()[(n_ts // 2):],
        )
        l = loss(loss_arg, verbose=True)
        l.backward()
        opt.step()
        opt.zero_grad()

    if epoch % log_interval == 0:
        if swa and epoch > start_swa:
            map_to_channels = torch.eye(n_channels, device=device)[
                cov_swa.module.weight.squeeze().argmax(0)]
        else:
            map_to_channels = torch.eye(n_channels, device=device)[
                cov.weight.squeeze().argmax(0)]
        lh = map_to_channels[:(n_ts // 2)].t() @ atlas_ref.weight['cortex_L']
        rh = map_to_channels[(n_ts // 2):].t() @ atlas_ref.weight['cortex_R']
        with torch.no_grad():
            atlas_plot.preweight['cortex_L'][:] = lh
            atlas_plot.preweight['cortex_R'][:] = rh
        views = (
            'medial', 'lateral'
        )
        plotter = fsLRAtlasParcels(atlas_plot)
        plotter(
            cmap=network_cmap,
            views=views,
            save=f'{results}/epoch-{epoch:08}_cmap-network'
        )
