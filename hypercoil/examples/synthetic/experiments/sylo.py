#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Sylo experiments
~~~~~~~~~~~~~~~~
Simple experiments using sylo modules.
"""
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from functools import partial
from typing import Optional
from matplotlib.pyplot import close
from hypercoil.engine.paramutil import PyTree, Tensor
from hypercoil.functional import delete_diagonal
from hypercoil.loss.nn import (
    MSELoss,
    UnilateralLoss,
    ModularityLoss,
)
from hypercoil.loss.scheme import (
    LossScheme,
    LossApply,
    LossArgument,
    UnpackingLossArgument,
)
from hypercoil.examples.synthetic.scripts.sylo import (
    synthesise_lowrank_block,
    plot_templates,
    plot_outcome,
    plot_conn,
    SyloShallowAutoencoder
)


def shallow_autoencoder_experiment(
    max_epoch: int = 50001,
    lr: float = 1e-3,
    recombinator_nonnegative_l2_nu: float = 0.1,
    nonnegative_l2_nu: float = 0.01,
    modularity_nu: float = 0.2,
    n_nodes: int = 100,
    rank: int = 4,
    save: Optional[str] = None,
    *,
    key: int,
):
    key = jax.random.PRNGKey(key)
    key_d, key_m, key_l = jax.random.split(key, 3)

    A = synthesise_lowrank_block(n_nodes, key=key_d)
    target = A
    conn = target.reshape((1, 1, n_nodes, n_nodes))
    target = delete_diagonal(target)
    plot_conn(A, save=f'{save}_ref.png')

    n_filters = rank

    model = SyloShallowAutoencoder(
        n_channels=1,
        n_filters=n_filters,
        dim=n_nodes,
        mix_bias=False,
        key=key_m
    )

    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    losses = []

    mod_softmax = partial(jax.nn.softmax, axis=-1)

    loss = LossScheme([
        LossApply(
            UnilateralLoss(
                name='RecombinatorPositive',
                nu=recombinator_nonnegative_l2_nu),
            apply=lambda arg: -arg.recomb.weight
        ),
        LossApply(
            ModularityLoss(
                name='Modularity',
                nu=modularity_nu,),
            apply=lambda arg: UnpackingLossArgument(
                A=target,
                Q=mod_softmax(arg.sylo.weight[0].squeeze().T),
                theta=(jnp.eye(n_filters) * arg.recomb.weight)
            )),
        LossApply(
            UnilateralLoss(
                name='SyloPositive',
                nu=nonnegative_l2_nu),
            apply=lambda arg: -arg.sylo.weight[0]
        ),
        LossApply(
            MSELoss(
                name='MSE',
                nu=1.0),
            apply=lambda arg: UnpackingLossArgument(
                Y_hat=arg.input,
                Y=arg.target
            )
        )
    ])

    saves = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150]
    saves += list(range(200, 1000, 100))
    saves += list(range(1000, 5000, 200))
    saves += list(range(5000, 10000, 500))
    saves += list(range(10000, max_epoch, 1000))

    def forward(
        model: PyTree,
        X: Tensor,
        target: Tensor,
        loss: LossScheme,
        *,
        key: int,
    ):
        key_m, key_l = jax.random.split(key)
        Y = model(X, key=key_m)
        arg = LossArgument(
            input=Y,
            target=target,
            sylo=model.sylo,
            recomb=model.rcmb,
        )
        loss_epoch, meta = loss(arg, key=key_l)
        return loss_epoch, meta

    for e in range(max_epoch):
        key_l = jax.random.fold_in(key_l, e)
        (loss_epoch, meta), grad = eqx.filter_jit(eqx.filter_value_and_grad(
            forward,
            has_aux=True,
        ))(model, conn, target, loss, key=key_l)
        updates, opt_state = opt.update(
            eqx.filter(grad, eqx.is_inexact_array),
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)

        losses += [loss_epoch.item()]
        if e in saves:
            print(f'[ Epoch {e} | Loss {loss_epoch} ]')
            for k, v in meta.items():
                print(f'{k}: {v.value:.4f}')
            plot_templates(
                model=model.sylo,
                X=conn,
                n_filters=n_filters,
                save=f'{save}_templates{e:08}.png'
            )
            plot_outcome(
                model=model,
                X=conn,
                save=f'{save}_outcome{e:08}.png'
            )
            close('all')


def main():
    from hypercoil.examples.synthetic.experiments.run import (
        run_layer_experiments
    )
    run_layer_experiments('sylo')


if __name__ == '__main__':
    main()
