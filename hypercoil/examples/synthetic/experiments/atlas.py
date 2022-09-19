#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlas experiments
~~~~~~~~~~~~~~~~~
Simple experiments on parcellated ground truth datasets.
"""
import torch
import numpy as np
from matplotlib.pyplot import close
from hypercoil.synth.atlas import (
    hard_atlas_example,
    hard_atlas_homologue,
    soft_atlas_example,
    hierarchical_atlas_example,
    plot_atlas,
    plot_hierarchical,
    embed_data_in_atlas,
    get_model_matrices
)
from hypercoil.functional import sym2vec
from hypercoil.loss import (
    Compactness,
    LogDetCorr,
    Entropy,
    Equilibrium,
    SecondMoment,
    LossScheme,
    LossApply,
    LossArgument,
    UnpackingLossArgument
)
from hypercoil.functional import corr
from hypercoil.functional.sphere import euclidean_conv


def atlas_experiment(
    parcellation='hard',
    homologue_parcellation=False,
    max_epoch=10000,
    lr=0.01,
    log_interval=100,
    seed=None,
    supervised=False,
    entropy_nu=0.5,
    logdet_nu=0.1,
    secondmoment_nu=1000,
    compactness_nu=1,
    equilibrium_nu=100,
    compactness_floor=5,
    save=None,
    image_dim=25,
    latent_dim=100,
    time_dim=300,
    parcel_count=9,
):
    if parcellation == 'hard':
        A = hard_atlas_example(d=image_dim)
    elif parcellation == 'hard2':
        A = hard_atlas_homologue(d=image_dim)
        parcellation = 'hard'
    elif parcellation == 'soft':
        A = soft_atlas_example(d=image_dim, c=parcel_count, seed=seed)
    elif parcellation == 'hierarchical':
        X, A = hierarchical_atlas_example(
            d=image_dim,
            seed=seed,
            t=time_dim,
            latent_dim=latent_dim
        )
    else:
        raise ValueError(f'Unrecognised parcellation string: {parcellation}')

    if parcellation == 'hierarchical':
        ts_reg = X.view(image_dim * image_dim, -1)
        ref, ts = None, ts_reg
    else:
        X, ts_reg = embed_data_in_atlas(
            A,
            parc=parcellation,
            t=time_dim,
            atlas_dim=parcel_count,
            signal_dim=latent_dim
        )
        ref, ts = get_model_matrices(A, X, parc=parcellation)

    if homologue_parcellation:
        Ah = hard_atlas_homologue(d=image_dim)
        Xh, ts_regh = embed_data_in_atlas(
            Ah,
            parc=parcellation,
            ts_reg=ts_reg,
            t=time_dim,
            atlas_dim=parcel_count
        )
        refh, tsh = get_model_matrices(Ah, Xh)
        plot_atlas(
            refh,
            d=image_dim,
            saveh=f'{save}hard-ref.png',
            saves=f'{save}soft-ref.png'
        )
    elif parcellation == 'hierarchical':
        plot_hierarchical(
            A, save=f'{save}hard-ref.png'
        )
    else:
        plot_atlas(
            ref,
            d=image_dim,
            saveh=f'{save}hard-ref.png',
            saves=f'{save}soft-ref.png'
        )


    Y = torch.FloatTensor(np.corrcoef(ts_reg))

    ax_x = torch.arange(image_dim).view(1, -1).tile(image_dim, 1)
    ax_y = torch.arange(image_dim).view(-1, 1).tile(1, image_dim)
    coor = torch.stack((
        ax_x.view(image_dim * image_dim),
        ax_y.view(image_dim * image_dim)
    ))


    if homologue_parcellation:
        model = euclidean_conv(
            ref.clone().transpose(-1, -2),
            coor.transpose(-1, -2),
            scale=3
        ).transpose(-1, -2)
        X = tsh
    else:
        model = 0.1 * torch.rand(parcel_count, image_dim * image_dim)
        X = ts
    model.requires_grad = True

    opt = torch.optim.Adam(params=[model], lr=lr)
    loss = torch.nn.MSELoss()

    reg = LossScheme([
        LossScheme([
            Compactness(nu=compactness_nu,
                        floor=compactness_floor,
                        coor=coor),
            Entropy(nu=entropy_nu, axis=-2),
            Equilibrium(nu=equilibrium_nu),
        ], apply=lambda arg: arg.model),
        LossApply(
            SecondMoment(nu=secondmoment_nu, skip_normalise=True),
            apply=lambda arg: UnpackingLossArgument(
                weight=arg.model,
                data=arg.x
            )
        ),
        LossApply(
            LogDetCorr(nu=logdet_nu),
            apply=lambda arg: arg.y
        )
    ])


    losses = []


    for epoch in range(max_epoch):
        mnorm = torch.softmax(model, 0)
        ts_parc = mnorm @ X
        corrmat = corr(ts_parc)
        arg = LossArgument(model=mnorm, x=X, y=ts_parc)
        if epoch == 0:
            print('Statistics at train start:')
            reg(arg, verbose=True)
        loss_epoch = 0
        if supervised:
            loss_epoch = loss(sym2vec(corr(ts_parc)), sym2vec(Y))
        loss_epoch = loss_epoch + reg(arg)
        loss_epoch.backward()
        losses += [loss_epoch.detach().item()]
        opt.step()
        model.grad.zero_()
        if epoch % log_interval == 0:
            print(f'[ Epoch {epoch} | Loss {loss_epoch} ]')
            plot_atlas(
                mnorm,
                d=image_dim,
                c=parcel_count,
                saveh=f'{save}hard-{epoch:08}.png',
                saves=f'{save}soft-{epoch:08}.png'
            )
            close('all')


def main():
    from hypercoil.synth.experiments.run import run_layer_experiments
    run_layer_experiments('atlas')


if __name__ == '__main__':
    main()
