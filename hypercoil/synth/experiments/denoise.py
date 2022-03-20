#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Denoising experiments
~~~~~~~~~~~~~~~~~~~~~
Simple experiments in artefact removal.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import orth
from hypercoil.functional.cov import (
    corr,
    conditionalcorr
)
from hypercoil.nn.select import (
    LinearCombinationSelector,
    EliminationSelector
)
from hypercoil.loss import (
    QCFC,
    NormedLoss
)
from hypercoil.synth.denoise import (
    synthesise_artefact,
    plot_all,
    plot_select
)
from hypercoil.synth.mix import (
    synthesise_mixture
)


def model_selection_experiment(
    model='combination',
    l1_nu=0,
    lr=0.01,
    max_epoch=100,
    log_interval=5,
    time_dim=1000,
    observed_dim=20,
    latent_dim=30,
    subject_dim=100,
    artefact_dim=20,
    correlated_artefact=False,
    spatial_heterogeneity=False,
    subject_heterogeneity=False,
    noise_scale=2,
    jitter=(0.1, 0.5, 1.5),
    include=(1, 1, 1),
    lp=0.3,
    tol=0,
    tol_sig=0.1,
    seed=None,
    orthogonalise=False,
    batch_size=None,
    save=None
):
    np.random.seed(seed)
    X = synthesise_mixture(
        time_dim=time_dim,
        observed_dim=observed_dim,
        latent_dim=latent_dim,
        subject_dim=subject_dim,
        lp=lp,
        seed=seed
    )
    artefactseed = seed
    if artefactseed is not None: artefactseed += 1
    N, artefact, NL = synthesise_artefact(
        time_dim=time_dim,
        observed_dim=artefact_dim,
        latent_dim=latent_dim,
        subject_dim=subject_dim,
        correlated_artefact=correlated_artefact,
        seed=artefactseed,
        lp=lp,
        jitter=jitter,
        include=include,
        spatial_heterogeneity=spatial_heterogeneity,
        subject_heterogeneity=subject_heterogeneity,
        noise_scale=noise_scale
    )
    X = torch.FloatTensor(X)
    N = torch.FloatTensor(N)
    NL = torch.FloatTensor(NL).view(1, -1)
    XN = X + torch.FloatTensor(artefact)
    if orthogonalise:
        for i, a in enumerate(artefact):
            artefact[i] = orth(a.T).T


    uniq_idx = torch.triu_indices(*(corr(XN)[0].shape), 1)

    modeltype = model
    if model == 'combination':
        model = LinearCombinationSelector(model_dim=3, n_columns=artefact_dim)
    elif model == 'elimination':
        model = EliminationSelector(n_columns=artefact_dim)
        #print(model.preweight, model.postweight)
    elif model == 'combelim':
        model = torch.nn.Sequential(
            LinearCombinationSelector(
                model_dim=artefact_dim,
                n_columns=artefact_dim
            ),
            EliminationSelector(n_columns=artefact_dim)
        )
        model[0].weight.requires_grad = False
        model[0].weight[:] = (
            torch.eye(artefact_dim) + 0.1 * torch.rand(artefact_dim, artefact_dim)
        )
        model[0].weight.requires_grad = True
    elif model == 'elimcomb':
        model = torch.nn.Sequential(
            EliminationSelector(n_columns=artefact_dim),
            LinearCombinationSelector(
                model_dim=artefact_dim,
                n_columns=artefact_dim
            )
        )
    if batch_size is None:
        batch_size = subject_dim
    loss = QCFC(tol=tol, tol_sig=tol_sig)
    reg = NormedLoss(nu=l1_nu, p=1, axis=-1)
    opt = torch.optim.Adam(params=model.parameters(), lr=lr)


    plot_all(
        corr(X),
        n_subj=subject_dim,
        cor=True,
        save=f'{save}-ref.png'
    )


    cor = corr(XN)
    print(f'[ QC-FC at train start: {loss(cor, NL)} ]')


    losses = []
    scores = []
    n_zero = -1


    elim = model
    if modeltype != 'combination':
        if modeltype == 'combelim': elim = model[1]
        if modeltype == 'elimcomb': elim = model[0]


    for epoch in range(max_epoch):
        batch_index = torch.LongTensor(np.random.permutation(subject_dim)[:batch_size])
        regs = model(N[batch_index])
        #TODO: We're making spike regressors in the elimination model. Not
        # necessarily what we want to be doing.
        regs = regs + 0.001 * torch.eye(regs.shape[-2], regs.shape[-1])
        cor = conditionalcorr(XN[batch_index], regs)
        cors = cor[:, uniq_idx[0], uniq_idx[1]]
        loss_epoch = (
            loss(cors, NL[:, batch_index])
            + reg(elim.weight)
        )
        loss_epoch.backward()
        score = ((corr(torch.tensor(X)) - conditionalcorr(XN, model(N)).detach()) ** 2).mean()
        losses += [loss_epoch.detach().numpy()]
        scores += [score]
        opt.step()
        model.zero_grad()

        if epoch % log_interval == 0:
            print(f'[ Epoch {epoch} | Loss {loss_epoch} | Score {score} ]')
            if modeltype != 'elimination':
                plot_all(
                    conditionalcorr(XN, model(N)).detach().numpy(),
                    n_subj=subject_dim,
                    cor=True,
                    save=f'{save}-{epoch:06}.png'
                )
            elif modeltype != 'combination':
                plot_select(elim, save=f'{save}-weight{epoch:06}.png')
            plt.close('all')

    plt.figure(figsize=(6, 6))
    plt.plot(scores)
    plt.ylabel('SSE Score')
    plt.xlabel('Epoch')
    plt.savefig(f'{save}-score.png', bbox_inches='tight')


def main():
    from hypercoil.synth.experiments.run import run_layer_experiments
    run_layer_experiments('denoise')


if __name__ == '__main__':
    main()
