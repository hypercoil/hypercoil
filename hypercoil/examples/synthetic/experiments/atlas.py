#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlas experiments
~~~~~~~~~~~~~~~~~
Simple experiments on parcellated ground truth datasets.
"""
import jax
import jax.numpy as jnp
import optax
import distrax
import equinox as eqx
from typing import Optional, Union
from matplotlib.pyplot import close
from hypercoil.engine.paramutil import _to_jax_array
from hypercoil.examples.synthetic.scripts.atlas import (
    hard_atlas_example,
    hard_atlas_homologue,
    soft_atlas_example,
    hierarchical_atlas_example,
    plot_atlas,
    plot_hierarchical,
    embed_data_in_atlas,
    get_model_matrices,
)
from hypercoil.functional import sym2vec, corr_kernel
from hypercoil.init.base import DistributionInitialiser
from hypercoil.loss.nn import (
    MSELoss,
    CompactnessLoss,
    GramLogDeterminantLoss,
    EntropyLoss,
    EquilibriumLoss,
    SecondMomentLoss,
)
from hypercoil.loss.scheme import (
    LossScheme,
    LossApply,
    LossArgument,
    LossReturn,
    UnpackingLossArgument,
)
from hypercoil.functional import corr
from hypercoil.functional.sphere import euclidean_conv
from hypercoil.init.mapparam import ProbabilitySimplexParameter
from hypercoil.nn import AtlasLinear


def atlas_experiment(
    parcellation: str = 'hard',
    homologue_parcellation: bool = False,
    max_epoch: int = 10000,
    lr: float = 0.01,
    log_interval: int = 100,
    supervised: bool = False,
    entropy_nu: float = 0.5,
    logdet_nu: float = 0.1,
    secondmoment_nu: int = 1000,
    compactness_nu: int = 1,
    equilibrium_nu: int = 100,
    compactness_floor: int = 5,
    save: Optional[str] = None,
    image_dim: int = 25,
    latent_dim: int = 100,
    time_dim: int = 300,
    parcel_count: int = 9,
    *,
    key: Union['jax.random.PRNGKey', int],
):
    if isinstance(key, int):
        key = jax.random.PRNGKey(key)
    if parcellation == 'hard':
        A = hard_atlas_example(d=image_dim)
    elif parcellation == 'hard2':
        A = hard_atlas_homologue(d=image_dim)
        parcellation = 'hard'
    elif parcellation == 'soft':
        A = soft_atlas_example(d=image_dim, c=parcel_count, key=key)
    elif parcellation == 'hierarchical':
        X, A = hierarchical_atlas_example(
            d=image_dim,
            t=time_dim,
            latent_dim=latent_dim,
            key=key,
        )
    else:
        raise ValueError(f'Unrecognised parcellation string: {parcellation}')

    if parcellation == 'hierarchical':
        ts_reg = X.reshape((image_dim * image_dim, -1))
        ref, ts = None, ts_reg
    else:
        key = jax.random.split(key, 1)[0]
        X, ts_reg = embed_data_in_atlas(
            A,
            parc=parcellation,
            t=time_dim,
            atlas_dim=parcel_count,
            signal_dim=latent_dim,
            key=key,
        )
        ref, ts = get_model_matrices(A, X, parc=parcellation)

    if homologue_parcellation:
        Ah = hard_atlas_homologue(d=image_dim)
        Xh, ts_regh = embed_data_in_atlas(
            Ah,
            parc=parcellation,
            ts_reg=ts_reg,
            t=time_dim,
            atlas_dim=parcel_count,
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

    if supervised:
        Y = jnp.corrcoef(ts_reg)
    else:
        Y = None

    ax_x = jnp.tile(jnp.arange(image_dim).reshape((1, -1)), (image_dim, 1))
    ax_y = jnp.tile(jnp.arange(image_dim).reshape((-1, 1)), (1, image_dim))
    coor = jnp.stack((
        ax_x.reshape(image_dim * image_dim),
        ax_y.reshape(image_dim * image_dim)
    ))

    key_m = jax.random.split(key, 1)[0]
    model = AtlasLinear(
        n_labels={'all': parcel_count},
        n_locations={'all': image_dim * image_dim},
        key=key_m,
        normalisation=None,
    )
    if homologue_parcellation:
        weight = euclidean_conv(
            ref.copy().swapaxes(-1, -2),
            coor.swapaxes(-1, -2),
            scale=3
        ).swapaxes(-1, -2)
        model = eqx.tree_at(lambda m: m.weight['all'], model, weight)
        X = tsh
    else:
        model = DistributionInitialiser.init(
            model,
            distribution=distrax.Uniform(0, 1),
            where='weight$all',
            key=key_m)
        X = ts
    model = ProbabilitySimplexParameter.map(
        model, where='weight$all', axis=-2)

    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    loss = MSELoss()

    reg = LossScheme([
        LossScheme([
            CompactnessLoss(
                nu=compactness_nu,
                name='Compactness',
                floor=compactness_floor,
                coor=coor,
                radius=None),
            EntropyLoss(
                nu=entropy_nu,
                name='Entropy',
                axis=-2),
            EquilibriumLoss(nu=equilibrium_nu, name='Equilibrium'),
        ], apply=lambda arg: arg.model),
        LossApply(
            SecondMomentLoss(
                nu=secondmoment_nu,
                name='SecondMoment'),
            apply=lambda arg: UnpackingLossArgument(
                weight=arg.model,
                X=arg.x
            )
        ),
        LossApply(
            GramLogDeterminantLoss(
                nu=logdet_nu,
                name='LogDetCorr',
                op=corr_kernel,
                psi=1e-3,
                xi=1e-3),
            apply=lambda arg: arg.y
        )
    ])

    losses = []

    def forward(model, X, Y=None, *, key):
        key_m, key_l = jax.random.split(key, 2)
        ts_parc = model(X, key=key_m)
        arg = LossArgument(
            model=_to_jax_array(model.weight['all']),
            x=X, y=ts_parc
        )
        loss_epoch = 0
        loss_meta = {}
        if Y is not None:
            loss_epoch = loss(sym2vec(corr(ts_parc)), sym2vec(Y))
            loss_meta = {loss.name: LossReturn(value=loss_epoch, nu=loss.nu)}
        reg_epoch, reg_meta = reg(arg, key=key_l)
        return loss_epoch + reg_epoch, {**loss_meta, **reg_meta}


    for epoch in range(max_epoch):
        key = jax.random.split(key, 1)[0]
        (total, meta), grad = eqx.filter_jit(eqx.filter_value_and_grad(
            forward,
            has_aux=True
        ))(model, X, Y, key=key)
        losses += [total]
        updates, opt_state = opt.update(
            eqx.filter(grad, eqx.is_inexact_array),
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)
        if jnp.isinf(total) or jnp.isnan(total):
            assert 0
        if epoch % log_interval == 0:
            print(f'[ Epoch {epoch} | Loss {total} ]')
            plot_atlas(
                _to_jax_array(model.weight['all']),
                d=image_dim,
                c=parcel_count,
                saveh=f'{save}hard-{epoch:08}.png',
                saves=f'{save}soft-{epoch:08}.png',
            )
            close('all')


def main():
    from hypercoil.examples.synthetic.experiments.run import (
        run_layer_experiments
    )
    run_layer_experiments('atlas')


if __name__ == '__main__':
    main()
