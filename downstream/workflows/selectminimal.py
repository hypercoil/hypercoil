#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Model selection
~~~~~~~~~~~~~~~
Minimal model selection workflow.
"""
import click
import torch
import pathlib
import pandas as pd
import webdataset as wds
from functools import partial
from hypercoil.data.transforms import Normalise
from hypercoil.data.functional import window_map
from hypercoil.data.collate import gen_collate, extend_and_bind
from hypercoil.engine import (
    Epochs,
    LossArchive,
    ModelArgument,
    UnpackingModelArgument
)
from hypercoil.functional import conditionalcorr, sym2vec
from hypercoil.functional.domainbase import Identity
from hypercoil.init.freqfilter import FreqFilterSpec
from hypercoil.loss import (
    ReducingLoss, QCFC, SoftmaxEntropy, LossScheme, LossApply
)
from hypercoil.loss.batchcorr import qcfc_loss
from hypercoil.nn.interpolate import HybridInterpolate
from hypercoil.nn.freqfilter import FrequencyDomainFilter
from hypercoil.nn.select import (
    BOLDPredict, QCPredict, ResponseFunctionLinearSelector
)
from hypercoil.nn.window import WindowAmplifier


fs = 0.72
n_ts = 400
lr = 0.005
wd = 1e-3
momentum=0.9
train_dir = '/mnt/andromeda/Data/HCP_parcels_wds/spl-{000..005}_shd-{000000..000004}.tar'
val_dir = '/mnt/andromeda/Data/HCP_parcels_wds/spl-{006..007}_shd-{000000..000004}.tar'
confounds_path = '/tmp/confounds/sub-{}_ses-{}_run-{}.pt'
confounds_path_orig = '/mnt/andromeda/Data/HCP_S1200/data/{}/MNINonLinear/Results/rfMRI_REST{}_{}/Confound_Regressors.tsv'
dtype = torch.float
save_dev = 'cpu'
device = 'cuda:0'
cols = ['framewise_displacement', 'std_dvars']
basis_functions = [
    #lambda x : torch.ones_like(x),
    lambda x: x,
    #lambda x: torch.log(torch.abs(x) + 1e-8),
    #torch.exp
]


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


def model_cfg(model_dim, n_response_functions, window_size, device):
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
    selector = ResponseFunctionLinearSelector(
        model_dim=model_dim,
        n_columns=57,
        leak=0.01,
        n_response_functions=n_response_functions,
        softmax=False,
        device=device
    )
    return interpol, bpfft, selector


def loss_cfg(max_epoch, resume_epoch):
    epochs = Epochs(max_epoch)
    qcfc = QCFC(nu=1, name='QC-FC')
    entropy = SoftmaxEntropy(nu=10, name='Entropy')

    loss_train = LossScheme([
        #LossApply(entropy, apply=(lambda arg: arg.weight)),
        LossApply(
            qcfc,
            apply=(lambda arg: UnpackingModelArgument(FC=arg.fc, QC=arg.qc)))
    ])
    qcfc_val = LossScheme([LossApply(
        QCFC(nu=1, name='QC-FC_Val'),
        apply=(lambda arg: UnpackingModelArgument(FC=arg.fc, QC=arg.qc))
    )])
    loss_tape = LossArchive(epochs=epochs)
    loss_train.register_sentry(loss_tape)
    qcfc_val.register_sentry(loss_tape)
    return epochs, loss_train, qcfc_val, loss_tape


def model_forward(model, bold, confounds, mask, qc, lossfn, selector):
    bold, confmodel, confs = model(bold, confounds, mask)
    cor = conditionalcorr(bold, confmodel)
    fc = sym2vec(cor)
    arg = ModelArgument(
        fc=fc,
        qc=qc.nanmean(1).unsqueeze(0),
        weight=selector.weight['lin']
    )
    loss = lossfn(arg, verbose=True)
    return loss, confmodel, bold, confs


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


@click.command()
@click.option('-t', '--train-dir', required=True)
@click.option('-v', '--val-dir', required=True)
@click.option('-o', '--out', required=True)
@click.option('-d', '--model-dim', required=True, type=int)
@click.option('-b', '--batch-size', required=True, type=int)
@click.option('--batch-size-val', default=None, type=int)
@click.option('-c', '--device', default='cuda:0')
@click.option('-w', '--window-size', default=200, type=int)
@click.option('-x', '--augmentation-factor', default=5, type=int)
@click.option('-r', '--n-response-functions', default=5, type=int)
@click.option('--lr', default=0.005, type=float)
@click.option('--wd', default=1e-3, type=float)
@click.option('--momentum', default=0.9, type=float)
@click.option('--max-epoch', default=201, type=int)
@click.option('--resume-epoch', default=-1, type=int)
@click.option('--saved-state', default=None, type=str)
def main(
    train_dir, val_dir, out, device, model_dim, batch_size,
    batch_size_val, window_size, augmentation_factor,
    n_response_functions, lr, max_epoch, wd, momentum,
    resume_epoch, saved_state
):
    if batch_size_val is None:
        batch_size_val = batch_size
    dl, dl_val = data_cfg(
        train_dir=train_dir,
        val_dir=val_dir,
        batch_size=batch_size,
        batch_size_val=batch_size_val
    )
    interpol, bpfft, selector = model_cfg(
        model_dim=model_dim,
        n_response_functions=n_response_functions,
        window_size=window_size,
        device=device
    )
    epochs, loss_train, qcfc_val, loss_tape = loss_cfg(max_epoch, resume_epoch)

    def model(bold, confs, mask, interpol, bpfft, selector):
        assert mask.dtype == torch.bool
        #ms = (mask.sum(-1).squeeze())
        #if torch.any(ms == 0):
        #    print(ms)
        bold = interpol(bold.unsqueeze(1), mask).squeeze()
        bold = bpfft(bold).squeeze()
        confs = interpol(confs.unsqueeze(1), mask).squeeze()
        confs = bpfft(confs).squeeze()
        confmodel = selector(confs)
        return bold, confmodel, confs

    model = partial(
        model,
        interpol=interpol,
        bpfft=bpfft,
        selector=selector
    )
    #opt = torch.optim.Adam(lr=lr, params=selector.parameters())
    train_model(dl=dl, dl_val=dl_val, epochs=epochs, selector=selector,
                model=model, loss_train=loss_train, qcfc_val=qcfc_val,
                loss_tape=loss_tape, window_size=window_size,
                augmentation_factor=augmentation_factor, out=out)


def train_model(dl, dl_val, epochs, selector, model,
                loss_train, qcfc_val, loss_tape,
                window_size, augmentation_factor, out):
    opt = torch.optim.SGD(
        lr=lr, momentum=momentum, weight_decay=wd,
        params=selector.parameters())

    wndw = WindowAmplifier(
        window_size=window_size,
        augmentation_factor=augmentation_factor
    )

    val_iter = iter(dl_val)
    for epoch in epochs:
        print(f'[ Epoch {epoch} ]')
        if epoch > 2:
            print(loss_tape.archive['QC-FC'][-3:],
                  loss_tape.archive['QC-FC_Val'][-3:])
        dl_iter = iter(dl)
        selector.train()
        #qcfc.tol = 0.1 * torch.rand(1).item()
        for _ in range(10):
            bold, confounds, qc, nanmask = ingest_data(dl_iter, wndw)
            loss, _, _, _ = model_forward(
                model, bold, confounds,
                nanmask.unsqueeze(1),
                qc, loss_train, selector)

            if torch.any(torch.isnan(loss)):
                assert 0

            loss.backward()
            opt.step()
            opt.zero_grad()


        refqcfc = 0
        with torch.no_grad():
            selector.eval()
            for _ in range(10):
                bold, confounds, qc, nanmask = ingest_data(val_iter, wndw)
                loss, confmodel, bold, confs = model_forward(
                    model, bold, confounds,
                    nanmask.unsqueeze(1),
                    qc, qcfc_val, selector)
                reffc = sym2vec(conditionalcorr(bold, confs[:, 8].unsqueeze(1)))
                refqcfc += qcfc_loss(QC=qc.nanmean(1).unsqueeze(0), FC=reffc).mean().item()
            print(f'  (Reference : {refqcfc / 10})')
            if epoch % 10 == 0:
                torch.save(selector.state_dict(), f'{out}_epoch-{epoch}_state.pt')


if __name__ == '__main__':
    main()
