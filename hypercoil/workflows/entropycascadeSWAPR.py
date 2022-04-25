#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Entropy cascade
~~~~~~~~~~~~~~~
Entropy cascade parcellation learning algorithm.
"""
import os
import torch
import click
import templateflow.api as tflow
from functools import partial
from pkg_resources import resource_filename as pkgrf
from hypercoil.functional import (
    spherical_geodesic,
    cmass_coor,
    residualise
)
from hypercoil.engine import (
    Epochs,
    MultiplierCascadeSchedule,
    MultiplierDecaySchedule,
    WeightDecayMultiStepSchedule,
    SWA,
    SWAPR,
    LossArchive
)
from hypercoil.loss import (
    LossScheme,
    LossApply,
    LossArgument,
    UnpackingLossArgument,
    Compactness,
    Entropy,
    Equilibrium,
    HemisphericTether,
    VectorDispersion,
    SecondMoment,
    LogDetCorr
)
from hypercoil.nn import AtlasLinear
from hypercoil.init.atlas import DirichletInitSurfaceAtlas
from hypercoil.engine.terminal import ReactiveTerminal
from hypercoil.viz.surf import (
    fsLRAtlasParcels,
    fsLRAtlasMaps
)
from hypercoil.data.functional import window_map, identity
from hypercoil.data.transforms import (
    Compose,
    PolynomialDetrend,
    Normalise,
    NaNFill
)
from hypercoil.data.wds import torch_wds


def configure_model(n_labels, device='cuda:0', lr=0.05, wd=0, saved_state=None):
    atlas_template = pkgrf(
        'hypercoil',
        'viz/resources/nullexample.nii'
    )
    dirichlet = DirichletInitSurfaceAtlas(
        cifti_template=atlas_template,
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
            'cortex_L': n_labels,
            'cortex_R': n_labels,
            'subcortex': 0
        },
        conc=100.,
        dtype=torch.float,
        device=device
    )
    atlas = AtlasLinear(dirichlet, mask_input=False,
                        domain=dirichlet.init['_all'].domain,
                        dtype=torch.float, device=device)

    atlas_swa = torch.optim.swa_utils.AveragedModel(atlas)

    lh_coor = atlas.coors[dirichlet.compartments['cortex_L'][atlas.mask]].t()
    rh_coor = atlas.coors[dirichlet.compartments['cortex_R'][atlas.mask]].t()

    lh_mask = dirichlet.compartments['cortex_L'][atlas.mask].unsqueeze(0)
    rh_mask = dirichlet.compartments['cortex_R'][atlas.mask].unsqueeze(0)

    opt = torch.optim.Adam(params=atlas.parameters(), lr=lr, weight_decay=wd)

    if saved_state is not None:
        state_dict = torch.load(saved_state)
        atlas.load_state_dict(state_dict['model'])
        opt.load_state_dict(state_dict['opt'])
        atlas_swa.load_state_dict(state_dict['swa_model'])
    return atlas, atlas_swa, opt, lh_coor, rh_coor, lh_mask, rh_mask


def configure_timing(
    atlas, lh_coor, rh_coor,
    max_epoch=5001,
    reactive_slice=5,
    resume_epoch=-1
):
    epochs = Epochs(max_epoch)
    epochs.set(resume_epoch)

    dispersion_nu = MultiplierCascadeSchedule(
        epochs=epochs, base=10,
        transitions={(80, 120): 2, (200, 300): 0.5}
    )
    determinant_nu = MultiplierCascadeSchedule(
        epochs=epochs, base=.005,
        transitions={(80, 120): .001, (200, 400): .0001}
    )
    equilibrium_nu = MultiplierDecaySchedule(
        epochs=epochs, base=1e6,
        transitions={
            (200, 500): (1e6, 25000),
            (500, 1500): (1e7, 1e7),
            (1500, 2000): (3e7, 3e6),
            (2000, 2500): (1e7, 1e6),
            (2500, 3000): (3e6, 3e5),
            (3000, 3500): (3e6, 3e5),
            (3500, 4000): (1e6, 1e5),
            (4000, 4500): (1e6, 1e5),
            (4500, 5000): (1e6, 1e5)}
    )
    entropy_nu = MultiplierCascadeSchedule(
        epochs=epochs, base=0.1,
        transitions={
            (100, 200): 0.0001,
            (500, 800): 0.1,
            (1500, 1505): 0.5,
            (2000, 2005): 1.0,
            (2500, 2505): 1.5,
            (3000, 3005) : 2.0,
            (3500, 3505): 2.5,
            (4000, 4005): 3.0,
            (4500, 4505): 4.0}
    )
    compactness_nu = MultiplierCascadeSchedule(
        epochs=epochs, base=2,
        transitions={
            (80, 120): 5,
            (400, 800): 20,
            (2500, 2505): 30,
            (4500, 4505): 40}
    )
    tether_nu = MultiplierCascadeSchedule(
        epochs=epochs, base=.2,
        transitions={(2650, 2700): 1}
    )
    second_moment_nu = MultiplierDecaySchedule(
        epochs=epochs, base=5,
        transitions={
            (1500, 1750): (10, 5),
            (1750, 2000): (5, 5),
            (2000, 2250): (10, 5),
            (2250, 2500): (5, 5),
            (2500, 2750): (10, 5),
            (2750, 3000): (5, 5),
            (3000, 3250): (7, 5),
            (3250, 3500): (5, 5),
            (3500, 3750): (7, 5),
            (3750, 4000): (5, 5),
            (4000, 4250): (7, 5),
            (4250, 5000): (5, 5),
        }
    )

    loss = LossScheme((
        LossScheme((
            Entropy(nu=entropy_nu, name='LeftHemisphereEntropy'),
            Equilibrium(nu=equilibrium_nu, name='LeftHemisphereEquilibrium'),
            Compactness(nu=compactness_nu, radius=100, coor=lh_coor,
                        name='LeftHemisphereCompactness')
        ), apply=lambda arg: arg.lh),
        LossScheme((
            Entropy(nu=entropy_nu, name='RightHemisphereEntropy'),
            Equilibrium(nu=equilibrium_nu, name='RightHemisphereEquilibrium'),
            Compactness(nu=compactness_nu, radius=100, coor=rh_coor,
                        name='RightHemisphereCompactness')
        ), apply=lambda arg: arg.rh),
        LossApply(
            HemisphericTether(nu=0.2, name='InterHemisphericTether'),
            apply=lambda arg: UnpackingLossArgument(
                lh=arg.lh,
                rh=arg.rh,
                lh_coor=arg.lh_coor,
                rh_coor=arg.rh_coor)
        ),
        LossApply(
            VectorDispersion(nu=dispersion_nu, metric=spherical_geodesic,
                             name='LeftHemisphereDispersion'),
            apply=lambda arg: cmass_coor(arg.lh, arg.lh_coor, 100).t()),
        LossApply(
            VectorDispersion(nu=dispersion_nu, metric=spherical_geodesic,
                             name='RightHemisphereDispersion'),
            apply=lambda arg: cmass_coor(arg.rh, arg.rh_coor, 100).t()),
        LossApply(
            LogDetCorr(nu=determinant_nu, psi=0.01, xi=0.01),
            apply=lambda arg: arg.ts)
    ))
    terminal_L = ReactiveTerminal(
        loss=SecondMoment(nu=second_moment_nu,
                          standardise=False,
                          skip_normalise=True,
                          name='LeftHemisphereSecondMoment'),
        slice_target='data',
        slice_axis=-1,
        max_slice=reactive_slice,
        pretransforms={'weight': atlas.domain.image}
    )
    terminal_R = ReactiveTerminal(
        loss=SecondMoment(nu=second_moment_nu,
                          standardise=False,
                          skip_normalise=True,
                          name='RightHemisphereSecondMoment'),
        slice_target='data',
        slice_axis=-1,
        max_slice=reactive_slice,
        pretransforms={'weight': atlas.domain.image}
    )

    loss_tape = LossArchive(epochs=epochs)
    loss.register_sentry(loss_tape)
    terminal_L.register_sentry(loss_tape)
    terminal_R.register_sentry(loss_tape)

    return epochs, loss, terminal_L, terminal_R, loss_tape


def configure_dataset(data_path, batch_size, buffer_size, window_size):
    wndw = partial(
        window_map,
        keys=('images', 'confounds', 'tmask'),
        window_length=window_size
    )
    maps = {
        'images' : identity,
        'confounds' : identity,
        'tmask' : identity,
        't_r' : identity,
        'task' : identity,
    }
    ds = torch_wds(
        data_path,
        keys=maps,
        batch_size=batch_size,
        shuffle=buffer_size,
        map=wndw
    )
    polybasis = torch.stack([
        torch.arange(500) ** i
        for i in range(3)
    ])
    return ds, polybasis


def transform_sample(sample, atlas, basis, device='cuda:0'):
    normalise = Normalise()
    ##TODO: we're just taking the first elements instead of masking...
    X = sample[0][:, :atlas.mask.sum(), :].float()
    gs = X.mean(-2)
    regs = torch.cat((
        basis.tile(X.shape[0], 1, 1),
        gs.unsqueeze(-2)
    ), -2).to(device)
    X = X.to(device)
    data = residualise(X, regs, driver='gels')
    data = normalise(data)
    return data


def compartmentalise_data(data, lh_mask, rh_mask):
    shape_L = (data.shape[0], lh_mask.sum(), -1)
    shape_R = (data.shape[0], rh_mask.sum(), -1)
    mask_L = lh_mask.tile(data.shape[0], 1)
    mask_R = rh_mask.tile(data.shape[0], 1)
    data_L = data[mask_L].view(*shape_L)
    data_R = data[mask_R].view(*shape_R)
    return data_L, data_R


def entropy_cascade(epochs, out_root, ds, atlas, atlas_swa, opt,
                    loss, terminal_L, terminal_R, n_labels, device,
                    lh_mask, rh_mask, lh_coor, rh_coor, loss_tape,
                    data_interval, log_interval, save_interval, basis,
                    resume_epoch):
    scheduler_lr = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=opt,
        milestones=(500, 1000, 2500),
        gamma=0.45,
        verbose=True)
    scheduler_swa = torch.optim.swa_utils.SWALR(
        optimizer=opt,
        swa_lr=0.05)
    scheduler_lr.last_epoch = resume_epoch - 1
    scheduler_swa.last_epoch = resume_epoch - 1
    start_swa = 1600
    swa = SWAPR(
        epochs=epochs,
        swa_start=start_swa,
        swa_model=atlas_swa,
        swa_scheduler=scheduler_swa,
        model=atlas,
        scheduler=scheduler_lr,
        revolve_epochs=(2050, 2550, 3050, 3550, 4050, 4550),
        device=device
    )
    schedule_wd = WeightDecayMultiStepSchedule(
        epochs=epochs,
        steps=[(2500, 1e-5)],
        param_groups=[opt.param_groups[0]]
    )
    modal_cmap = pkgrf(
        'hypercoil',
        'viz/resources/cmap_modal.nii'
    )
    network_cmap = pkgrf(
        'hypercoil',
        'viz/resources/cmap_network.nii'
    )
    cmaps = [modal_cmap, network_cmap]
    views = ('medial', 'lateral')
    atlas_basename = (
        f'desc-entropyCascade_res-dTSC{n_labels:06}'
    )
    atlas_name = f'{atlas_basename}_atlas'
    results = f'{out_root}/{atlas_name}'
    os.makedirs(results, exist_ok=True)
    no_data = True

    for epoch in epochs:
        if epoch % data_interval == 0 or no_data:
            #TODO: fix this data looping, use a dataloader
            for sample in ds:
                data = transform_sample(sample, atlas, basis, device)
                break
            no_data = False

        data_L, data_R = compartmentalise_data(data, lh_mask, rh_mask)
        ts = atlas(data)
        arg = LossArgument(
            ts=ts,
            lh=atlas.weight['cortex_L'],
            rh=atlas.weight['cortex_R'],
            lh_coor=lh_coor, rh_coor=rh_coor
        )
        arg_L = {'data': data_L, 'weight': atlas.preweight['cortex_L']}
        arg_R = {'data': data_R, 'weight': atlas.preweight['cortex_R']}

        out = terminal_L(arg=arg_L)
        print(f'- {terminal_L.loss} : {out}')
        #if (epoch % log_interval == 0): print(f'{terminal_L.loss} : {out}')
        out = terminal_R(arg=arg_R)
        print(f'- {terminal_R.loss} : {out}')
        #if (epoch % log_interval == 0): print(f'{terminal_R.loss} : {out}')
        #out = loss(arg, verbose=(epoch % log_interval == 0))
        out = loss(arg, verbose=True)
        print(f'[ Epoch {epoch} | Total loss {out} ]\n')
        out.backward()
        opt.step()
        opt.zero_grad()

        if epoch % log_interval == 0:
            plotter = fsLRAtlasParcels(atlas)
            plotter(
                cmap=network_cmap,
                views=views,
                save=f'{results}/epoch-{epoch:08}'
            )
            if epoch > start_swa:
                plotter = fsLRAtlasParcels(swa.swa_model.module)
                plotter(
                    cmap=network_cmap,
                    views=views,
                    save=f'{results}/desc-SWA_epoch-{epoch:08}'
                )

        if epoch % save_interval == 0:
            torch.save({
                'model': atlas.state_dict(),
                'opt': opt.state_dict(),
                'swa_model': swa.swa_model.state_dict()
            }, f'{results}/params-{epoch:08}.tar')
            df = loss_tape.data
            df.to_csv(f'{results}/loss_tape-{epoch:08}.tsv',
                      sep='\t', index=None)

    all_views = (
        'dorsal', 'ventral',
        'posterior', 'anterior',
        'medial', 'lateral'
    )
    plotter = fsLRAtlasParcels(atlas)
    plotter(
        cmap=modal_cmap,
        views=all_views,
        save=f'{results}/{atlas_basename}_cmap-modal'
    )
    plotter(
        cmap=network_cmap,
        views=all_views,
        save=f'{results}/{atlas_basename}_cmap-network'
    )


@click.command()
@click.option('-o', '--out-root', required=True, type=str)
@click.option('-d', '--data', 'data_path', required=True, type=str)
@click.option('-l', '--labels', 'n_labels', default=200, type=int)
@click.option('-c', '--device', default='cuda:0', type=str)
@click.option('--max-epoch', default=5001, type=int)
@click.option('--saved-state', default=None, type=str)
@click.option('--saved-archive', default=None, type=str)
@click.option('--resume-epoch', default=-1, type=int)
@click.option('--log-interval', default=25, type=int)
@click.option('--save-interval', default=250, type=int)
@click.option('--data-interval', default=5, type=int)
@click.option('--batch-size', default=3, type=int)
@click.option('--buffer-size', default=10, type=int)
@click.option('--window-size', default=500, type=int)
@click.option('--reactive-slice', default=5, type=int)
def main(out_root, data_path, n_labels, device, max_epoch,
         saved_state, saved_archive, resume_epoch, log_interval,
         save_interval, data_interval, batch_size, buffer_size,
         window_size, reactive_slice):
    lr = 0.05
    wd = 0
    (atlas, atlas_swa, opt,
     lh_coor, rh_coor,
     lh_mask, rh_mask) = configure_model(
        n_labels=n_labels,
        device=device,
        lr=lr,
        wd=wd,
        saved_state=saved_state
    )
    (epochs, loss,
     terminal_L,
     terminal_R,
     loss_tape) = configure_timing(
        atlas=atlas,
        lh_coor=lh_coor,
        rh_coor=rh_coor,
        max_epoch=max_epoch,
        resume_epoch=resume_epoch,
        reactive_slice=reactive_slice,
    )
    ds, basis = configure_dataset(
        data_path,
        batch_size,
        buffer_size,
        window_size
    )
    entropy_cascade(
        epochs=epochs,
        out_root=out_root,
        ds=ds,
        atlas=atlas,
        atlas_swa=atlas_swa,
        opt=opt,
        loss=loss,
        terminal_L=terminal_L,
        terminal_R=terminal_R,
        n_labels=n_labels,
        device=device,
        lh_mask=lh_mask,
        rh_mask=rh_mask,
        lh_coor=lh_coor,
        rh_coor=rh_coor,
        loss_tape=loss_tape,
        data_interval=data_interval,
        log_interval=log_interval,
        save_interval=save_interval,
        basis=basis,
        resume_epoch=resume_epoch
    )


if __name__ == '__main__':
    main()
