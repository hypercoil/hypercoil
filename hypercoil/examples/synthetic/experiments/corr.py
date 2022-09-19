#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Correlation experiments
~~~~~~~~~~~~~~~~~~~~~~~
Simple experiments using covariance modules. Basically, state detection.
"""
import torch
import numpy as np
from matplotlib.pyplot import close
from hypercoil.functional.cov import corr, pairedcorr
from hypercoil.init.domain import (
    Identity,
    MultiLogit
)
from hypercoil.functional.matrix import sym2vec, vec2sym
from hypercoil.init.base import (
    DistributionInitialiser
)
from hypercoil.nn.cov import UnaryCovariance
from hypercoil.loss import (
    Entropy,
    Equilibrium,
    NormedLoss,
    SmoothnessPenalty,
    SymmetricBimodalNorm,
    VectorDispersion,
    LossScheme,
    LossApply,
    LossArgument
)
from hypercoil.synth.corr import (
    synthesise_state_transition,
    synthesise_state_markov_chain,
    get_transitions,
    plot_states,
    plot_state_ts,
    plot_state_transitions,
    correlation_alignment,
    kmeans_init
)
from hypercoil.synth.experiments.overfit_plot import overfit_and_plot_progress


def state_detection_experiment(
    lr=0.01,
    max_epoch=1000,
    log_interval=25,
    time_dim=1000,
    latent_dim=7,
    observed_dim=100,
    n_states=2,
    state_timing='equipartition',
    test_state=-1,
    seed=None,
    save=None
):
    if seed is not None: torch.manual_seed(seed)
    np.random.seed(seed)
    X, Y, target = synthesise_state_transition(
        time_dim=time_dim,
        latent_dim=latent_dim,
        observed_dim=observed_dim,
        n_states=n_states,
        state_timing=state_timing,
        seed=seed
    )
    max_tol_score = np.sqrt(.01 * time_dim)
    init = DistributionInitialiser(
        distr=torch.distributions.Uniform(0.45, 0.55),
        domain=Identity()
    )
    model = UnaryCovariance(
        dim=time_dim,
        estimator=corr,
        max_lag=0,
        init=init
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    #opt = torch.optim.SGD(model.parameters(), lr=1, momentum=0.2)

    loss = torch.nn.MSELoss()
    reg = LossScheme([
        SymmetricBimodalNorm(nu=0.005, modes=(0., 1.)),
        SmoothnessPenalty(nu=0.05)
    ], apply=lambda x: x.weight)


    X = torch.Tensor(X)
    Y = torch.Tensor(Y[test_state])
    target = target[:, test_state]


    overfit_and_plot_progress(
        out_fig=save, model=model, optim=opt, reg=reg, loss=loss,
        max_epoch=max_epoch, X=X, Y=Y, target=target,
        log_interval=log_interval, seed=seed
    )


    target = torch.Tensor(target)
    solution = model.weight.squeeze().detach()

    score = loss(target, solution)
    # This is the score if every guess were exactly 0.1 from the target.
    # (already far better than random chance)
    assert(score < max_tol_score)

    # Fewer than 1 in 10 are more than 0.1 from target
    score = ((target - solution).abs() > 0.1).sum().item()
    assert(score < time_dim // 10)


def unsupervised_state_detection_experiment(
    lr=0.01,
    max_epoch=1001,
    log_interval=20,
    smoothness_nu=5,
    symbimodal_nu=1,
    entropy_nu=0.01,
    equilibrium_nu=5000,
    dist_nu=1,
    within_nu=0.002,
    between_nu=0.5,
    time_dim=1000,
    subject_dim=100,
    latent_dim=10,
    observed_dim=30,
    state_weight=1,
    subject_weight=1,
    n_states=6,
    batch_size=None,
    transition_matrix=None,
    begin_states=None,
    seed=None,
    save=None
):
    if seed is not None: torch.manual_seed(seed)
    np.random.seed(seed)
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
        seed=seed
    )

    plot_state_ts(srcmat, save=f'{save}-stateTS.png')

    if subject_dim == 1:
        transitions = get_transitions(srcmat)
        active_state = active_state.squeeze()
        init = DistributionInitialiser(
            distr=torch.distributions.Uniform(0.45, 0.55),
            domain=MultiLogit(axis=0)
        )
        model = UnaryCovariance(
            dim=time_dim,
            estimator=corr,
            out_channels=n_states,
            init=init
        )
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        #model_x_y : tuple of
        # (model, x, correlation)
        loss = LossScheme([
            LossScheme([
                SmoothnessPenalty(nu=smoothness_nu),
                Entropy(nu=entropy_nu, axis=0),
                Equilibrium(nu=equilibrium_nu)
            ], apply=lambda arg: arg.model.weight),
            LossApply(
                SymmetricBimodalNorm(nu=symbimodal_nu, modes=(-1, 1)),
                apply=lambda arg: arg.cor
            ),
            LossApply(
                VectorDispersion(nu=dist_nu),
                apply=lambda arg: sym2vec(arg.cor)
            )
        ])

        X = torch.tensor(observed_ts)

        losses = []

        for epoch in range(max_epoch):
            #TODO: using the contrast against time-averaged connectivity.
            # Is this really the better approach?
            cor = model(X) - corr(X)
            arg = LossArgument(model=model, X=X, cor=cor)
            if epoch == 0:
                loss(arg, verbose=True)
            loss_epoch = loss(arg)

            loss_epoch.backward()
            losses += [loss_epoch.detach().item()]
            opt.step()
            model.zero_grad()
            if epoch % log_interval == 0:
                print(f'[ Epoch {epoch} | Loss {loss_epoch} ]')
                plot_state_transitions(
                    model.weight.detach().squeeze().t().numpy(),
                    transitions=transitions,
                    save=f'{save}_transitionTS-{epoch:08}.png'
                )
                plot_states(
                    model(X).detach().numpy(),
                    save=f'{save}_states-{epoch:08}.png'
                )
                close('all')

        groundtruth = corr(
            torch.tensor(observed_ts),
            weight=torch.tensor(active_state).t()
        )
        realigned = correlation_alignment(
            X=sym2vec(groundtruth),
            X_hat=sym2vec(model(X)),
            n_states=n_states
        )
        plot_states(
            vec2sym(realigned.detach()).numpy() + np.eye(observed_dim),
            save=f'{save}_states-ref.png'
        )

    else:
        #TODO: One day we must come back and clean up this monstrosity of
        # code.
        transitions = np.where(np.diff(srcmat[0]) != 0)[0]

        #i_g_x_y : tuple of
        # (individual model, group model, x, correlation)
        # Unfortunately Python is . . . not great with functional
        # programming.
        loss = LossScheme([
            LossScheme([
                SmoothnessPenalty(nu=smoothness_nu),
                Entropy(nu=entropy_nu, axis=0),
                Equilibrium(nu=equilibrium_nu)
            ], apply=lambda arg: arg.individual),
            LossApply(
                SymmetricBimodalNorm(nu=symbimodal_nu, modes=(-1, 1)),
                apply=lambda arg: arg.y
            ),
            LossApply(
                VectorDispersion(nu=dist_nu),
                apply=lambda arg: sym2vec(arg.y)
            ),
            LossApply(
                VectorDispersion(nu=between_nu, name='ClusterBetween'),
                apply=lambda arg: arg.group
            ),
            LossApply(
                NormedLoss(nu=within_nu, p=1, name='ClusterWithin'),
                apply=lambda arg: sym2vec(arg.y) - arg.group,
            )
        ])

        X = torch.FloatTensor(observed_ts).unsqueeze(-3)
        best_states = corr(
            X,
            weight=torch.tensor(active_state.swapaxes(-1, -2)).unsqueeze(-2)
        ).mean(0)

        losses = []
        measures = []


        individual_model = torch.log(
            torch.distributions.Uniform(0.45, 0.55).sample(
                (subject_dim, n_states, 1, time_dim)
            )
        )
        #group_model = sym2vec(
        #    corr(X).mean(0) + 0.05 * torch.rand(
        #        (n_states, observed_dim, observed_dim)
        #    )
        #).float()
        group_model = kmeans_init(
            X=X,
            n_states=n_states,
            time_dim=time_dim,
            subject_dim=subject_dim
        )
        individual_model.requires_grad = True
        group_model.requires_grad = True

        static = corr(X)
        subject_specific = (static - static.mean(0))
        subject_specific_vec = sym2vec(subject_specific)


        opt = torch.optim.Adam(params=[individual_model, group_model], lr=lr)

        for epoch in range(max_epoch):
            if batch_size is not None:
                batch_index = torch.LongTensor(
                    np.random.permutation(subject_dim)[:batch_size]
                )
            else:
                batch_index = torch.arange(subject_dim)

            individual_mnorm = torch.softmax(
                individual_model[batch_index],
                axis=-3
            )
            correl = corr(X[batch_index], weight=individual_mnorm)
            cor = correl - subject_specific[batch_index]
            arg = LossArgument(
                individual=individual_mnorm,
                group=group_model,
                x=X, y=cor)
            loss_epoch = loss(arg)

            loss_epoch.backward()
            losses += [loss_epoch.detach().item()]
            opt.step()
            individual_model.grad.zero_()
            group_model.grad.zero_()

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
                    group_model,
                    sym2vec(corr(torch.FloatTensor(state_mix)))
                )
                maxim = statecorr.max()
                measure = (
                    statecorr.amax(0) -
                    statecorr.median(0)[0]
                ).mean().detach().item()
                measures += [measure]

                print(f'[ Epoch {epoch} | Loss {loss_epoch} | '
                      f'Measure {measure} | Maximum {maxim} ]')
                loss(arg, verbose=True)

                recovered_states = corr(
                    X,
                    weight=torch.softmax(individual_model, axis=-3)
                ).mean(0)
                plot_state_transitions(
                    individual_mnorm[0].detach().squeeze().t().numpy(),
                    transitions=transitions,
                    save=f'{save}_transitionTS-{epoch:08}.png',
                )
                plot_states(
                    (vec2sym(group_model).detach().numpy() +
                        np.eye(observed_dim)),
                    save=f'{save}_stateTemplates-{epoch:08}.png',
                    vpct=.9
                )
                plot_states(
                    recovered_states.detach().numpy(),
                    save=f'{save}_states-{epoch:08}.png',
                    vpct=.9
                )
                close('all')
        realigned_best = correlation_alignment(
            X=sym2vec(best_states).float(),
            X_hat=sym2vec(recovered_states).float(),
            n_states=n_states
        )
        groundtruth = corr(
            torch.FloatTensor(state_mix),
        )
        realigned_groundtruth = correlation_alignment(
            X=sym2vec(groundtruth).float(),
            X_hat=sym2vec(recovered_states).float(),
            n_states=n_states
        )
        plot_states(
            vec2sym(realigned_best).detach().numpy() + np.eye(observed_dim),
            save=f'{save}_states-best.png',
            vpct=.9
        )
        plot_states(
            (vec2sym(realigned_groundtruth).detach().numpy() +
                np.eye(observed_dim)),
            save=f'{save}_states-true.png'
        )


def main():
    from hypercoil.synth.experiments.run import run_layer_experiments
    run_layer_experiments('corr')


if __name__ == '__main__':
    main()
