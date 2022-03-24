#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Sylo experiments
~~~~~~~~~~~~~~~~
Simple experiments using sylo modules.
"""
import torch
import numpy as np
from functools import partial
from matplotlib.pyplot import close
from hypercoil.functional import delete_diagonal
from hypercoil.loss import (
    UnilateralNormedLoss,
    ModularityLoss,
    LossScheme,
    LossApply,
    LossArgument,
    UnpackingLossArgument
)
from hypercoil.synth.sylo import (
    synthesise_lowrank_block,
    plot_templates,
    plot_outcome,
    plot_conn,
    SyloShallowAutoencoder
)


def shallow_autoencoder_experiment(
    max_epoch=50001,
    lr=1e-3,
    recombinator_nonnegative_l2_nu=0.1,
    nonnegative_l2_nu=0.01,
    modularity_nu=0.2,
    n_nodes=100,
    rank=4,
    seed=None,
    save=None
):
    if seed is not None: torch.manual_seed(seed)
    np.random.seed(seed)

    A = synthesise_lowrank_block(n_nodes, seed=seed)
    target = torch.FloatTensor(A)
    conn = target.view(1, 1, n_nodes, n_nodes)
    target = delete_diagonal(target)
    plot_conn(A, save=f'{save}_ref.png')

    n_filters = rank

    sy_low = SyloShallowAutoencoder(
        n_channels=1,
        n_filters=n_filters,
        dim=n_nodes,
        mix_bias=False
    )

    opt = torch.optim.Adam(sy_low.parameters(), lr=lr)
    losses = []

    mod_softmax = partial(torch.softmax, dim=-1)

    loss = LossScheme([
        LossApply(
            UnilateralNormedLoss(nu=recombinator_nonnegative_l2_nu),
            apply=lambda arg: -arg.recomb.weight
        ),
        LossApply(
            ModularityLoss(nu=modularity_nu, affiliation_xfm=mod_softmax),
            apply=lambda arg: UnpackingLossArgument(
                A=target,
                C=arg.sylo.weight_L.squeeze().t(),
                L=(torch.eye(n_filters) * arg.recomb.weight)
            )),
        LossApply(
            UnilateralNormedLoss(nu=nonnegative_l2_nu),
            apply=lambda arg: -arg.sylo.weight_L
        ),
        LossApply(
            torch.nn.MSELoss(),
            apply=lambda arg: UnpackingLossArgument(
                input=arg.input,
                target=arg.target
            )
        )
    ])

    saves = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150]
    saves += list(range(200, 1000, 100))
    saves += list(range(1000, 5000, 200))
    saves += list(range(5000, 10000, 500))
    saves += list(range(10000, max_epoch, 1000))

    for e in range(max_epoch):
        opt.zero_grad()
        reconn = sy_low(conn)
        arg = LossArgument(
            input=reconn,
            target=target,
            sylo=sy_low.net[0],
            recomb=sy_low.net[-1],
        )
        if e == 0:
            loss(arg, verbose=True)
        loss_epoch = loss(arg)
        loss_epoch.backward()
        opt.step()
        loss_last = loss_epoch.detach().numpy()
        losses += [loss_last]
        if e in saves:
            print(f'[ Epoch {e} | Loss {loss_last} ]')
            plot_templates(
                layer=sy_low.net[0],
                X=conn,
                n_filters=n_filters,
                save=f'{save}_templates{e:08}.png'
            )
            plot_outcome(
                model=sy_low,
                X=conn,
                save=f'{save}_outcome{e:08}.png'
            )
            close('all')


def main():
    from hypercoil.synth.experiments.run import run_layer_experiments
    run_layer_experiments('sylo')


if __name__ == '__main__':
    main()
