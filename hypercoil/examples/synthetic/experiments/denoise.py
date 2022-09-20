#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Denoising experiments
~~~~~~~~~~~~~~~~~~~~~
Simple experiments in artefact removal.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import distrax
import optax
import matplotlib.pyplot as plt
from typing import Callable, Literal, Optional, Tuple
from scipy.linalg import orth
from hypercoil.engine.noise import (
    Diagonal
)
from hypercoil.engine.paramutil import _to_jax_array, PyTree, Tensor
from hypercoil.functional.cov import (
    corr,
    conditionalcorr
)
from hypercoil.functional.matrix import sym2vec
from hypercoil.init.base import (
    DistributionInitialiser,
)
from hypercoil.nn.confound import (
    LinearCombinationSelector,
    EliminationSelector,
)
from hypercoil.loss.nn import (
    QCFCLoss,
    NormedLoss,
)
from hypercoil.examples.synthetic.scripts.denoise import (
    synthesise_artefact,
    plot_all,
    plot_select,
    plot_norm_reduction
)
from hypercoil.examples.synthetic.scripts.mix import (
    synthesise_mixture
)


def model_selection_experiment(
    model: (
        Literal['combination', 'elimination', 'combelim', 'elimcomb']
    ) = 'combination',
    l1_nu: float = 0,
    lr: float = 0.01,
    max_epoch: int = 100,
    log_interval: int = 5,
    time_dim: int = 1000,
    observed_dim: int = 20,
    latent_dim: int = 30,
    subject_dim: int = 100,
    artefact_dim: int = 20,
    correlated_artefact: bool = False,
    spatial_heterogeneity: bool = False,
    subject_heterogeneity: bool = False,
    noise_scale: float = 2.,
    jitter: Tuple[float, float, float] = (0.1, 0.5, 1.5),
    include: Tuple[float, float, float] = (1, 1, 1),
    lp: float = 0.3,
    tol: float = 0,
    tol_sig: float = 0.1,
    orthogonalise: bool = False,
    batch_size: Optional[int] = None,
    save: Optional[str] = None,
    *,
    key: int,
):
    key = jax.random.PRNGKey(key)
    key_s, key_a, key_m, key_b, key_l = jax.random.split(key, 5)
    X = synthesise_mixture(
        time_dim=time_dim,
        observed_dim=observed_dim,
        latent_dim=latent_dim,
        subject_dim=subject_dim,
        lp=lp,
        key=key_s,
    )
    N, artefact, NL = synthesise_artefact(
        time_dim=time_dim,
        observed_dim=artefact_dim,
        latent_dim=latent_dim,
        subject_dim=subject_dim,
        correlated_artefact=correlated_artefact,
        lp=lp,
        jitter=jitter,
        include=include,
        spatial_heterogeneity=spatial_heterogeneity,
        subject_heterogeneity=subject_heterogeneity,
        noise_scale=noise_scale,
        key=key_a,
    )
    NL = NL.reshape((1, -1))
    XN = X + artefact
    if orthogonalise:
        for i, a in enumerate(artefact):
            artefact[i] = orth(a.T).T


    uniq_idx = jnp.triu_indices(*(corr(XN)[0].shape), 1)

    modeltype = model
    if model == 'combination':
        model = LinearCombinationSelector(
            model_dim=3,
            num_columns=artefact_dim,
            key=key_m)
    elif model == 'elimination':
        model = EliminationSelector(num_columns=artefact_dim, key=key_m)
        #print(model.preweight, model.postweight)
    elif model == 'combelim':
        key_mc, key_me = jax.random.split(key_m)
        model = eqx.nn.Sequential((
            LinearCombinationSelector(
                model_dim=artefact_dim,
                num_columns=artefact_dim,
                key=key_mc,
            ),
            EliminationSelector(num_columns=artefact_dim, key=key_me)
        ))
        init_distr = distrax.Uniform(0.9, 1.1)
        #TODO: we actually want off-diagonals here. But we don't use combelim,
        #      so whatever
        init_distr = Diagonal(init_distr, artefact_dim)
        DistributionInitialiser.init(
            model, distribution=init_distr, param_name='#0.weight')
        # model[0].weight[:] = (
        #     torch.eye(artefact_dim) + 0.1 * torch.rand(artefact_dim, artefact_dim)
        # )
    elif model == 'elimcomb':
        model = jax.nn.Sequential((
            EliminationSelector(n_columns=artefact_dim),
            LinearCombinationSelector(
                model_dim=artefact_dim,
                n_columns=artefact_dim
            )
        ))
    if batch_size is None:
        batch_size = subject_dim
    loss = QCFCLoss(
        name='QCFC',
        nu=1.,
        tol=tol,
        tol_sig=tol_sig)
    reg = NormedLoss(nu=l1_nu, p=1, axis=-1)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))


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


    def forward(
        model: PyTree,
        XN: Tensor,
        batch_index: Tensor,
        elim: PyTree,
        loss: Callable,
        reg: Callable,
    ) -> Tensor:
        regs = model(N[batch_index])
        #TODO: We're making spike regressors in the elimination model. Not
        # necessarily what we want to be doing.
        regs = regs + 0.001 * jnp.eye(regs.shape[-2], regs.shape[-1])
        cor = conditionalcorr(XN[batch_index], regs).squeeze()
        cors = sym2vec(cor)
        loss = loss(cors, NL[..., batch_index])
        if elim is not None:
            loss += reg(_to_jax_array(elim.weight))
        return loss


    for epoch in range(max_epoch):
        key_b = jax.random.split(key_b, 1)[0]
        key_l = jax.random.split(key_l, 1)[0]
        batch_index = jax.random.permutation(key_l, subject_dim)[:batch_size]

        if modeltype == 'combination':
            elim = None
        else:
            elim = model
            if modeltype == 'combelim': elim = model[1]
            if modeltype == 'elimcomb': elim = model[0]

        loss_epoch, grad = eqx.filter_value_and_grad(forward)(
            model, XN, batch_index, elim, loss, reg
        )
        regs = model(N)
        score = ((corr(X) - conditionalcorr(
            XN,
            regs + 0.001 * jnp.eye(regs.shape[-2], regs.shape[-1]))) ** 2
        ).mean()
        losses += [loss_epoch.item()]
        scores += [score]
        updates, opt_state = opt.update(
            eqx.filter(grad, eqx.is_inexact_array),
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)

        if epoch % log_interval == 0:
            print(f'[ Epoch {epoch} | Loss {loss_epoch} | Score {score} ]')
            if modeltype != 'elimination':
                plot_all(
                    conditionalcorr(XN, model(N)).squeeze(),
                    n_subj=subject_dim,
                    cor=True,
                    save=f'{save}-{epoch:06}.png'
                )
                plot_norm_reduction(
                    model.weight['lin'], save=f'{save}-norm{epoch:06}.png')
            elif modeltype != 'combination':
                plot_select(elim, save=f'{save}-weight{epoch:06}.png')
            plt.close('all')

    plt.figure(figsize=(6, 6))
    plt.plot(scores)
    plt.ylabel('SSE Score')
    plt.xlabel('Epoch')
    plt.savefig(f'{save}-score.png', bbox_inches='tight')


def main():
    from hypercoil.examples.synthetic.experiments.run import (
        run_layer_experiments
    )
    run_layer_experiments('denoise')


if __name__ == '__main__':
    main()
