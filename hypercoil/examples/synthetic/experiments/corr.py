#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Correlation experiments
~~~~~~~~~~~~~~~~~~~~~~~
Simple experiments using covariance modules. Basically, state detection.
"""
import jax
import jax.numpy as jnp
import distrax
import equinox as eqx
import optax
from functools import partial
from typing import Callable, Mapping, Optional, Tuple
from matplotlib.pyplot import close
from hypercoil.engine.paramutil import PyTree, Tensor, _to_jax_array
from hypercoil.functional.cov import corr, pairedcorr
from hypercoil.functional.kernel import linear_distance
from hypercoil.functional.matrix import sym2vec, vec2sym
from hypercoil.init.base import (
    DistributionInitialiser
)
from hypercoil.init.mapparam import ProbabilitySimplexParameter
from hypercoil.nn.cov import UnaryCovariance
from hypercoil.loss.nn import (
    MSELoss,
    EntropyLoss,
    EquilibriumLoss,
    NormedLoss,
    SmoothnessLoss,
    BimodalSymmetricLoss,
    DispersionLoss,
)
from hypercoil.loss.scalarise import (
    max_scalarise,
    meansq_scalarise,
    vnorm_scalarise,
)
from hypercoil.loss.scheme import (
    LossScheme,
    LossApply,
    LossArgument,
    UnpackingLossArgument,
)
from hypercoil.examples.synthetic.scripts.corr import (
    synthesise_state_transition,
    synthesise_state_markov_chain,
    get_transitions,
    plot_states,
    plot_state_ts,
    plot_state_transitions,
    correlation_alignment,
    kmeans_init
)
from hypercoil.examples.synthetic.experiments.overfit_plot import (
    overfit_and_plot_progress,
)


def state_detection_experiment(
    lr: float = 0.01,
    max_epoch: int = 1000,
    log_interval: int = 25,
    time_dim: int = 1000,
    latent_dim: int = 7,
    observed_dim: int = 100,
    n_states: int = 2,
    state_timing: Optional[str] = 'equipartition',
    test_state: int = -1,
    save: Optional[str] = None,
    *,
    key: int,
):
    key = jax.random.PRNGKey(key)
    key_d, key_m, key_l = jax.random.split(key, 3)
    X, Y, target = synthesise_state_transition(
        time_dim=time_dim,
        latent_dim=latent_dim,
        observed_dim=observed_dim,
        n_states=n_states,
        state_timing=state_timing,
        key=key_d,
    )
    max_tol_score = jnp.sqrt(.01 * time_dim)
    model = UnaryCovariance(
        dim=time_dim,
        estimator=corr,
        max_lag=0,
        key=key_m,
    )
    model = DistributionInitialiser.init(
        model, distribution=distrax.Uniform(0.45, 0.55), key=key_m,
    )
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    loss = MSELoss(
        name='MSE',
        nu=1.,)
    reg = LossScheme([
        BimodalSymmetricLoss(
            name='BimodalSymmetric',
            nu=0.005,
            modes=(0., 1.),
            scalarisation=vnorm_scalarise,),
        SmoothnessLoss(
            name='Smoothness',
            nu=0.05,
            scalarisation=vnorm_scalarise,),
    ], apply=lambda x: x.weight)


    X = jnp.array(X)
    Y = jnp.array(Y[test_state])
    target = target[:, test_state]


    overfit_and_plot_progress(
        out_fig=save, model=model, optim_state=opt_state, optim=opt, reg=reg,
        loss=loss, max_epoch=max_epoch, X=X, Y=Y, target=target,
        log_interval=log_interval, key=key_l,
    )


    target = jnp.array(target)
    solution = model.weight.squeeze()

    score = loss(target, solution)
    # This is the score if every guess were exactly 0.1 from the target.
    # (already far better than random chance)
    assert(score < max_tol_score)

    # Fewer than 1 in 10 are more than 0.1 from target
    score = (jnp.abs((target - solution)) > 0.1).sum().item()
    assert(score < time_dim // 10)


def unsupervised_state_detection_experiment(
    lr: float = 0.01,
    max_epoch: int = 1001,
    log_interval: int = 20,
    smoothness_nu: float = 5.,
    symbimodal_nu: float = 1.,
    entropy_nu: float = 0.01,
    equilibrium_nu: float = 5000.,
    dist_nu: float = 1.,
    within_nu: float = 0.002,
    between_nu: float = 0.5,
    time_dim: int = 1000,
    subject_dim: int = 100,
    latent_dim: int = 10,
    observed_dim: int = 30,
    state_weight: float = 1.,
    subject_weight: float = 1.,
    n_states: int = 6,
    batch_size: Optional[int] = None,
    transition_matrix: Optional[Tensor] = None,
    begin_states: Optional[Tensor] = None,
    save: Optional[str] = None,
    *,
    key: int,
):
    key = jax.random.PRNGKey(key)
    key_d, key_m, key_l = jax.random.split(key, 3)
    (
        observed_ts,
        srcmat,
        state_mix,
        active_state
    ) = synthesise_state_markov_chain(
        time_dim=time_dim,
        subject_dim=subject_dim,
        latent_dim=latent_dim,
        observed_dim=observed_dim,
        state_weight=state_weight,
        subject_weight=subject_weight,
        n_states=n_states,
        transition_matrix=transition_matrix,
        begin_states=begin_states,
        key=key_d
    )

    plot_state_ts(srcmat, save=f'{save}-stateTS.png')

    if subject_dim == 1:
        transitions = get_transitions(srcmat)
        active_state = active_state.squeeze()
        model = UnaryCovariance(
            dim=time_dim,
            estimator=corr,
            out_channels=n_states,
            key=key_m,
        )
        model = DistributionInitialiser.init(
            model,
            distribution=distrax.Uniform(0.45, 0.55),
            key=key_m,
        )
        model = ProbabilitySimplexParameter.map(model, axis=0)
        opt = optax.adam(learning_rate=lr)
        opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

        loss = LossScheme([
            LossScheme([
                SmoothnessLoss(
                    name='Smoothness',
                    nu=smoothness_nu,
                    scalarisation=partial(vnorm_scalarise, p=2, axis=-1),
                ),
                EntropyLoss(
                    name='Entropy',
                    nu=entropy_nu,
                    axis=0),
                EquilibriumLoss(
                    name='Equilibrium',
                    nu=equilibrium_nu)
            ], apply=lambda arg: _to_jax_array(arg.model.weight)),
            LossScheme([
                BimodalSymmetricLoss(
                    name='BimodalSymmetric',
                    nu=symbimodal_nu,
                    modes=(-1, 1),
                    scalarisation=meansq_scalarise,
                ),
                DispersionLoss(
                    name='Dispersion',
                    nu=dist_nu,
                    metric=linear_distance,
                )],
                apply=lambda arg: sym2vec(arg.cor)
            )
        ])

        X = jnp.array(observed_ts)

        losses = []

        def forward(
            model: PyTree,
            X: Tensor,
            loss: Callable,
            key: 'jax.random.PRNGKey'
        ) -> Tuple[PyTree, Mapping]:
            key_m, key_l = jax.random.split(key, 2)
            #TODO: using the contrast against time-averaged connectivity.
            # Is this really the better approach?
            cor = model(X, key=key_m) - corr(X)
            arg = LossArgument(model=model, X=X, cor=cor)
            loss_epoch, meta = loss(arg, key=key_l)
            return loss_epoch, meta

        for epoch in range(max_epoch):
            key_l = jax.random.fold_in(key_l, epoch)
            (loss_epoch, meta), grad = eqx.filter_value_and_grad(
                forward, has_aux=True
            )(model, X, loss, key=key_l)
            # for k, v in meta.items():
            #     print(f'{k}: {v.value:.4f}')
            updates, opt_state = opt.update(
                eqx.filter(grad, eqx.is_inexact_array),
                opt_state,
                eqx.filter(model, eqx.is_inexact_array),
            )
            model = eqx.apply_updates(model, updates)
            if epoch % log_interval == 0:
                print(f'[ Epoch {epoch} | Loss {loss_epoch} ]')
                plot_state_transitions(
                    _to_jax_array(model.weight).T,
                    transitions=transitions,
                    save=f'{save}_transitionTS-{epoch:08}.png'
                )
                plot_states(
                    model(X),
                    save=f'{save}_states-{epoch:08}.png'
                )
                close('all')

        groundtruth = corr(
            observed_ts,
            weight=active_state.T,
        )
        realigned = correlation_alignment(
            X=sym2vec(groundtruth),
            X_hat=sym2vec(model(X)),
            n_states=n_states,
        )
        plot_states(
            vec2sym(realigned) + jnp.eye(observed_dim),
            save=f'{save}_states-ref.png'
        )

    else:
        #TODO: One day we must come back and clean up this monstrosity of
        # code.
        transitions = jnp.where(jnp.diff(srcmat[0]) != 0)[0]

        loss = LossScheme([
            LossScheme([
                SmoothnessLoss(
                    name='Smoothness',
                    nu=smoothness_nu,
                    scalarisation=meansq_scalarise,
                    #scalarisation=partial(vnorm_scalarise, p=2, axis=-1),
                ),
                EntropyLoss(
                    name='Entropy',
                    nu=entropy_nu,
                    axis=0),
                EquilibriumLoss(
                    name='Equilibrium',
                    nu=equilibrium_nu,
                    instance_axes=(-3, -1),)
            ], apply=lambda arg: arg.individual),
            LossApply(
                BimodalSymmetricLoss(
                    name='BimodalSymmetric',
                    nu=symbimodal_nu,
                    modes=(-1, 1),
                    scalarisation=meansq_scalarise,),
                apply=lambda arg: arg.y
            ),
            LossApply(
                DispersionLoss(
                    name='Dispersion',
                    nu=dist_nu),
                apply=lambda arg: arg.y
            ),
            LossApply(
                DispersionLoss(
                    name='ClusterBetween',
                    nu=between_nu,),
                apply=lambda arg: arg.group
            ),
            LossApply(
                NormedLoss(
                    name='ClusterWithin',
                    nu=within_nu,
                    p=1,),
                apply=lambda arg: arg.y - arg.group,
            )
        ])

        X = observed_ts[..., None, :, :]
        best_states = corr(
            X,
            weight=active_state.swapaxes(-1, -2)[..., None, :]
        ).mean(0)

        losses = []
        measures = []


        key_mi, key_mg = jax.random.split(key_m, 2)
        individual_model = jnp.log(
            distrax.Uniform(0.45, 0.55).sample(
                seed=key_mi,
                sample_shape=(subject_dim, n_states, 1, time_dim)
            )
        )
        group_model = kmeans_init(
            X=X,
            n_states=n_states,
            time_dim=time_dim,
            subject_dim=subject_dim,
            key=key_mg,
        )

        static = corr(X)
        subject_specific = (static - static.mean(0))
        #subject_specific_vec = sym2vec(subject_specific)

        class DifferentiableClustering(eqx.Module):
            individual_model: Tensor
            group_model: Tensor
            subject_specific: Tensor

            def __call__(
                self,
                X: Tensor,
                batch_index: Tensor,
                *,
                key: 'jax.random.PRNGKey',
            ) -> Tensor:
                individual_mnorm = jax.nn.softmax(
                    self.individual_model[batch_index],
                    axis=-3
                )
                correl = corr(X[batch_index], weight=individual_mnorm)
                cor = correl - jax.lax.stop_gradient(
                    self.subject_specific[batch_index])
                return LossArgument(
                    individual=individual_mnorm,
                    group=self.group_model,
                    x=X,
                    y=sym2vec(cor)
                )

        model = DifferentiableClustering(
            individual_model=individual_model,
            group_model=group_model,
            subject_specific=subject_specific,
        )

        opt = optax.adam(learning_rate=lr)
        opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

        def forward(
            model: PyTree,
            X: Tensor,
            batch_index: Tensor,
            loss: Callable,
            *,
            key: 'jax.random.PRNGKey'
        ) -> Tuple[PyTree, Mapping]:
            key_m, key_l = jax.random.split(key, 2)
            arg = model(X, batch_index=batch_index, key=key_m)
            loss_epoch, meta = loss(arg, key=key_l)
            return loss_epoch, meta

        for epoch in range(max_epoch):
            key_l = jax.random.fold_in(key_l, epoch)
            if batch_size is not None:
                batch_index = jnp.random.permutation(
                    key_l, (subject_dim,))[:batch_size]
            else:
                batch_index = jnp.arange(subject_dim)

            (loss_epoch, meta), grad = eqx.filter_jit(eqx.filter_value_and_grad(
                forward,
                has_aux=True,
            ))(model, X, batch_index, loss, key=key_l)

            updates, opt_state = opt.update(
                eqx.filter(grad, eqx.is_inexact_array),
                opt_state,
                eqx.filter(model, eqx.is_inexact_array),
            )
            model = eqx.apply_updates(model, updates)

            """
            # If you dare attempt this without a good initialisation, growing
            # the lr might be of help. But it likely won't end well.
            lr_max = 0.01
            if epoch > 100:
                if (epoch + 1) % 100 == 0 and lr < lr_max:
                    lr *= 2
                    opt.param_groups[0]['lr'] = lr
                    print(f'Learning rate: {lr}')
                if (epoch + 1) % 100 == 0 and lr > lr_max:
                    lr = lr_max
                    print(f'Not really. Real learning rate: {lr}')
            """

            if epoch % log_interval == 0:

                statecorr = pairedcorr(
                    model.group_model,
                    sym2vec(corr(state_mix))
                )
                maxim = statecorr.max()
                measure = (
                    statecorr.max(0) -
                    jnp.median(statecorr, 0)[0]
                ).mean().item()
                measures += [measure]

                print(f'[ Epoch {epoch} | Loss {loss_epoch} | '
                      f'Measure {measure} | Maximum {maxim} ]')
                for k, v in meta.items():
                    print(f'{k}: {v.value:.4f}')

                individual_mnorm = jax.nn.softmax(
                    model.individual_model[batch_index],
                    axis=-3
                )
                recovered_states = corr(
                    X,
                    weight=individual_mnorm,
                ).mean(0)
                plot_state_transitions(
                    individual_mnorm[0].squeeze().swapaxes(-1, -2),
                    transitions=transitions,
                    save=f'{save}_transitionTS-{epoch:08}.png',
                )
                plot_states(
                    (vec2sym(model.group_model) + jnp.eye(observed_dim)),
                    save=f'{save}_stateTemplates-{epoch:08}.png',
                    vpct=.9
                )
                plot_states(
                    recovered_states,
                    save=f'{save}_states-{epoch:08}.png',
                    vpct=.9
                )
                close('all')
        realigned_best = correlation_alignment(
            X=sym2vec(best_states),
            X_hat=sym2vec(recovered_states),
            n_states=n_states
        )
        groundtruth = corr(state_mix)
        realigned_groundtruth = correlation_alignment(
            X=sym2vec(groundtruth),
            X_hat=sym2vec(recovered_states),
            n_states=n_states
        )
        plot_states(
            vec2sym(realigned_best) + jnp.eye(observed_dim),
            save=f'{save}_states-best.png',
            vpct=.9
        )
        plot_states(
            (vec2sym(realigned_groundtruth) +
                jnp.eye(observed_dim)),
            save=f'{save}_states-true.png'
        )


def main():
    from hypercoil.examples.synthetic.experiments.run import run_layer_experiments
    run_layer_experiments('corr')


if __name__ == '__main__':
    main()
