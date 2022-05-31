#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Covariance: state detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~
State detection covariance experiment. Hard codes everywhere until we have
time to clean it up.
"""
import os
import torch
import pathlib
import pandas as pd
import webdataset as wds
import templateflow.api as tflow
import matplotlib.pyplot as plt
from functools import partial
from pkg_resources import resource_filename as pkgrf
from hypercoil.data.transforms import Normalise
from hypercoil.data.collate import gen_collate, extend_and_bind
from hypercoil.engine import (
    Epochs, ModelArgument, UnpackingModelArgument, SWA
)
from hypercoil.engine.ephemeral import SGDEphemeral
from hypercoil.functional import corr, sym2vec, vec2sym
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
    VectorDispersion,
    NormedLoss,
    SmoothnessPenalty
)
from hypercoil.loss.jsdiv import JSDivergence
from hypercoil.nn.atlas import AtlasLinear
from hypercoil.nn.cov import UnaryCovariance
from hypercoil.nn.interpolate import HybridInterpolate
from hypercoil.nn.freqfilter import FrequencyDomainFilter
from hypercoil.nn.window import WindowAmplifier
from hypercoil.synth.corr import sliding_window_weight
from hypercoil.viz.surf import fsLRAtlasParcels


n_ts = 400
fs = 0.72
lr = 0.5
lr_template = 0.005
wd = 1e-4
momentum=0.9
batch_size = 100
batch_size_val = 200
window_size = 1200
sliding_window_size = 50
sliding_step_size = 25
device = 'cpu'
save_dev = 'cpu'
total_n_regs = 57
augmentation_factor = 1
n_channels = 3
log_interval = 1
entropy_nu = 0 # 0.2 # 1e-3
equilibrium_nu = 500000
dispersion_nu = 1e0 #0 # 1e-1 # 1e-2 # this is really just oiling the loss. likely unnecessary.
symbimodal_nu = 0 # 5
within_nu = 2e2 # 2e-5
between_nu = 2 # 5e-3
smoothness_nu = 10000 # 10
max_epoch = 51
resume_epoch = -1
instances_per_epoch = 100 # 500 # 5000
steps_per_batch = 1 # 10
swa = True
dtype = torch.float
template_init = f'/mnt/andromeda/Data/HCP_dynamics/centroids-{n_channels:03}.pt'
train_dir = '/mnt/andromeda/Data/HCP_parcels_wds/spl-{000..003}_shd-{000000..000004}.tar'
train_dir = '/mnt/andromeda/Data/HCP_parcels_wds/spl-000_shd-000001.tar'
val_dir = '/mnt/andromeda/Data/HCP_parcels_wds/spl-{004..007}_shd-{000000..000004}.tar'
confounds_path = '/tmp/confounds/sub-{}_ses-{}_run-{}.pt'
cols = ['framewise_displacement', 'std_dvars'] # we probably won't actually need these here
confounds_path_orig = (
    '/mnt/andromeda/Data/HCP_S1200/data/{}/MNINonLinear/'
    'Results/rfMRI_REST{}_{}/Confound_Regressors.tsv')
results = f'/tmp/statedetect_desc-{n_channels:04}'
#instance_dir = '/mnt/andromeda/Data/HCP_dynamics/instance_params'
instance_dir = f'{results}/instance_params'
reference_instance = 'sub-115825_task-REST_ses-2_run-LR'


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
            SmoothnessPenalty(nu=smoothness_nu),
            Entropy(nu=entropy_nu, axis=0),
            Equilibrium(nu=equilibrium_nu)
        ], apply=lambda arg: arg.weight),
        LossScheme([
            VectorDispersion(nu=dispersion_nu, name='SubjectDispersion'),
            SymmetricBimodal(nu=symbimodal_nu, modes=(-1, 1))
        ], apply=lambda arg: arg.states - arg.timeavg),
        LossApply(
            NormedLoss(nu=within_nu, p=2, name='ClusterWithin'),
            apply=lambda arg: arg.states - arg.template
        )
    ])
    loss_template = LossScheme([
        LossApply(
            VectorDispersion(nu=between_nu, name='ClusterBetween'),
            apply=lambda arg: (arg.template - arg.timeavg)
        ),
    ])
    return epochs, loss, loss_template


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
        nanmask.to(device=device),
        data['__key__']
    )


def preprocess(bold, confs, mask, interpol, bpfft, selector):
    with torch.no_grad():
        assert mask.dtype == torch.bool
        confs = selector @ confs
        bold = interpol(bold.unsqueeze(1), mask).squeeze()
        bold = bpfft(bold).squeeze()
        confs = interpol(confs.unsqueeze(1), mask).squeeze()
        confs = bpfft(confs).squeeze()

        #print(bold.shape, confs.shape)
        #theta = confs.transpose(-1, -2).pinverse() @ bold.transpose(-1, -2)
        #print(bold.shape, confs.shape, theta.shape)
        #bold = bold - theta.transpose(-1, -2) @ confs
        return residualise(bold, confs, driver='gels')
        return bold


def simple_parse(key):
    out = [e.split('-')[-1] for e in key.split('_')]
    return tuple(out)


def ingest_instance_weights(instance_dir, ids, opt):
    with torch.no_grad():
        instance_params = []
        instance_momentum_buffers = []
        for id in ids:
            sub, task, ses, run = simple_parse(id)
            instance_path = f'{instance_dir}/sub-{sub}_task-{task}_ses-{ses}_run-{run}.pt'
            if not os.path.exists(instance_path):
                param = 0.05 * torch.randn(n_channels, window_size, device=device)
                instance_params += [param]
                instance_momentum_buffers += [torch.zeros_like(param)]
            else:
                instance_data = torch.load(instance_path, map_location=device)
                instance_params += [instance_data['params']]
                instance_momentum_buffers += [instance_data['momentum_buffer']]
        instance_params = torch.stack(instance_params)
    instance_params.requires_grad = True
    if len(instance_momentum_buffers) > 0:
        instance_momentum_buffers = torch.stack(instance_momentum_buffers)
        opt.load_ephemeral(
            params=[instance_params],
            momentum_buffers=[instance_momentum_buffers]
        )
    else:
        opt.load_ephemeral(
            params=[instance_params]
        )
    return opt, instance_params


def write_instance_weights(
    instance_dir, ids, instance_params,
    instance_momentum_buffers):
    for i, id in enumerate(ids):
        sub, task, ses, run = simple_parse(id)
        instance_path = f'{instance_dir}/sub-{sub}_task-{task}_ses-{ses}_run-{run}.pt'
        instance_data = {
            'params' : instance_params[i],
            'momentum_buffer' : instance_momentum_buffers[i]
        }
        torch.save(instance_data, instance_path)


def template_cfg(sliding_window_size, sliding_step_size, window_size,
                 dl, wndw, interpol, bpfft, selector, n_channels,
                 n_batches=50, hypersphere=True):
    if not os.path.exists(template_init):
        from scipy.cluster.vq import kmeans
        renorm_term = 1
        if hypersphere:
            renorm_term = []
        sliding = sliding_window_weight(
            sliding_window_size,
            sliding_step_size,
            window_size)

        dynamic = []
        iter_dl = iter(dl)
        for _ in range(n_batches):
            bold, confounds, qc, nanmask, _ = ingest_data(iter_dl, wndw)
            denoised = preprocess(bold, confounds, nanmask,
                                  interpol, bpfft, selector)

            dynamic_conn = sym2vec(corr(denoised.unsqueeze(1),
                                        weight=sliding.unsqueeze(1)))
            if hypersphere:
                conn_norm = torch.linalg.norm(dynamic_conn, dim=-1, keepdim=True)
                dynamic_conn = dynamic_conn / conn_norm
                renorm_term += [conn_norm.mean()]
            dynamic += [dynamic_conn.view(-1, n_ts * (n_ts - 1) // 2)]
        dynamic = torch.cat(dynamic)
        if hypersphere:
            print(torch.tensor(renorm_term).mean())
            renorm_term = torch.tensor(renorm_term).mean()
        centroids, error = kmeans(dynamic, k_or_guess=n_channels)
        centroids = torch.tensor(centroids, device=device) * renorm_term
        torch.save(centroids, template_init)
    else:
        centroids = torch.load(template_init, map_location=device)
    template = centroids
    template.requires_grad = True
    return template

os.makedirs(results, exist_ok=True)
os.makedirs(instance_dir, exist_ok=True)

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
wndw = WindowAmplifier(
    window_size=window_size,
    augmentation_factor=augmentation_factor
)
epochs, loss, loss_template = loss_cfg(max_epoch, resume_epoch)
batches_per_epoch = instances_per_epoch // batch_size

template = template_cfg(
    sliding_window_size=sliding_window_size,
    sliding_step_size=sliding_step_size,
    window_size=window_size,
    dl=dl,
    wndw=wndw,
    interpol=interpol,
    bpfft=bpfft,
    selector=selector,
    n_channels=n_channels,
    n_batches=50
)
template_init = template.clone().detach()

dummy = torch.tensor(0)
opt = SGDEphemeral(params=[dummy], lr=lr, momentum=momentum, weight_decay=wd)
opt_template = torch.optim.SGD(params=[template], lr=lr_template, momentum=momentum, weight_decay=wd)

for epoch in epochs:
    iter_dl = iter(dl)
    print(f'\n[ Epoch {epoch} ]')
    timeavg_weight = 0
    running_timeavg = torch.zeros(n_ts, n_ts)
    if epoch % log_interval == 0:
        mean_weight = 0
        running_mean = torch.zeros((n_channels, n_ts, n_ts))
    for _ in range(batches_per_epoch):
        print(f'\n[ new batch ]')
        bold, confounds, qc, nanmask, key = ingest_data(iter_dl, wndw)
        with torch.no_grad():
            opt, instance_params = ingest_instance_weights(
                instance_dir=instance_dir,
                ids=key,
                opt=opt
            )
            denoised = preprocess(bold, confounds, nanmask, interpol, bpfft, selector)
        #print(instance_params[0])
        for _ in range(steps_per_batch):
            weight = torch.softmax(instance_params, 1)
            instance_states = corr(
                denoised.unsqueeze(1),
                weight=weight.unsqueeze(2)
            )
            timeavg = corr(denoised).mean(0)
            total_weight = timeavg_weight + batch_size
            running_timeavg = (
                timeavg_weight * running_timeavg +
                batch_size * timeavg
            ) / total_weight
            timeavg_weight = total_weight
            arg = ModelArgument(
                weight=weight,
                states=sym2vec(instance_states),
                template=template,
                timeavg=sym2vec(timeavg)
            )
            l = loss(arg, verbose=True)
            l.backward()
            instance_dict = opt.step()[instance_params]
            #print(instance_params[0])
            opt.zero_grad()
        #assert 0
        write_instance_weights(
            instance_dir=instance_dir,
            ids=key,
            instance_params=instance_params,
            instance_momentum_buffers=instance_dict['momentum_buffer']
        )
        opt.purge_ephemeral()
        if epoch % log_interval == 0:
            total_weight = mean_weight + batch_size
            running_mean = (
                mean_weight * running_mean +
                batch_size * instance_states.detach().mean(0)
            ) / total_weight
            mean_weight = total_weight
    arg = ModelArgument(
        template=template,
        timeavg=sym2vec(running_timeavg)
    )
    l_t = loss_template(arg, verbose=True)
    l_t.backward()
    opt_template.step()
    opt_template.zero_grad()
    #opt.zero_grad()

    if epoch % log_interval == 0:
        fig, (ax_state, ax_template, ax_init) = plt.subplots(
            3, n_channels, figsize=(4 * n_channels, 12)
        )
        for i in range(n_channels):
            ax_state[i].imshow(
                -running_mean[i].squeeze(),
                cmap='RdBu',
                vmin=-0.25,
                vmax=0.25)
            ax_state[i].set_xticks([])
            ax_state[i].set_yticks([])
            ax_template[i].imshow(
                -vec2sym(template[i].detach().clone().squeeze()),
                cmap='RdBu',
                vmin=-0.25,
                vmax=0.25)
            ax_template[i].set_xticks([])
            ax_template[i].set_yticks([])
            ax_init[i].imshow(
                -vec2sym(template_init[i].detach().clone()),
                cmap='RdBu',
                vmin=-0.25,
                vmax=0.25)
            ax_init[i].set_xticks([])
            ax_init[i].set_yticks([])
        save = f'{results}/desc-states_epoch-{epoch:08}.png'
        fig.savefig(save)

        ref_path = f'{instance_dir}/{reference_instance}.pt'
        if os.path.exists(ref_path):
            ref_param = torch.load(ref_path, map_location='cpu')['params'].detach().t()
        else:
            ref_param = torch.zeros_like(instance_params[0])
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        ax.plot(torch.softmax(ref_param, 1))
        ax.set_ylim(0, 1)
        ax.set_xlim(0, window_size)
        save = f'{results}/desc-param_epoch-{epoch:08}.png'
        fig.savefig(save)
