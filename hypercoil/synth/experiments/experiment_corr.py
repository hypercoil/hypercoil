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
from hypercoil.functional.cov import corr
from hypercoil.functional.domain import (
    Identity,
    MultiLogit
)
from hypercoil.functional.matrix import sym2vec, vec2sym
from hypercoil.init.base import (
    DistributionInitialiser
)
from hypercoil.nn.cov import UnaryCovariance
from hypercoil.reg import (
    Entropy,
    Equilibrium,
    SmoothnessPenalty,
    SymmetricBimodal,
    RegularisationScheme
)
from hypercoil.synth.corr import (
    synthesise_state_transition,
    synthesise_state_markov_chain,
    get_transitions,
    plot_states,
    plot_state_ts,
    plot_state_transitions,
    correlation_alignment
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
    reg = RegularisationScheme([
        SymmetricBimodal(nu=0.005, modes=(0., 1.)),
        SmoothnessPenalty(nu=0.05)
    ])


    X = torch.Tensor(X)
    Y = torch.Tensor(Y[test_state])
    target = target[:, test_state]
    #print(model(X))
    #raise Exception


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
    time_dim=1000,
    subject_dim=100,
    latent_dim=10,
    observed_dim=30,
    state_weight=1,
    subject_weight=1,
    n_states=6,
    transition_matrix=None,
    begin_states=None,
    seed=None,
    save=None
):
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

        reg_time = RegularisationScheme([
            SmoothnessPenalty(nu=smoothness_nu),
            Entropy(nu=entropy_nu, axis=0),
            Equilibrium(nu=equilibrium_nu)
        ])
        reg_corr = SymmetricBimodal(nu=symbimodal_nu, modes=(-1, 1))

        reg = (
            (reg_time, lambda model, X, Y: model.weight),
            (reg_corr, lambda model, X, Y: Y)
        )

        X = torch.tensor(observed_ts)

        losses = []

        for epoch in range(max_epoch):
            #TODO: using the contrast against time-averaged connectivity.
            # Is this really the better approach?
            cor = model(X) - corr(X)
            loss_epoch = 0
            for r, penalise in reg:
                loss_epoch += r(penalise(model, X, cor))

            #TODO: let's write a regulariser.
            corvex = sym2vec(cor)
            loss_epoch -= dist_nu * torch.cdist(corvex, corvex, p=1).mean()

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
        pass


if __name__ == '__main__':
    import os, hypercoil
    results = os.path.abspath(f'{hypercoil.__file__}/../results')

    """
    print('\n-----------------------------------------')
    print('Experiment 1: State Identification')
    print('-----------------------------------------')
    os.makedirs(f'{results}/corr_expt-stateident', exist_ok=True)
    state_detection_experiment(
        lr=0.01,
        max_epoch=500,
        log_interval=25,
        time_dim=1000,
        latent_dim=20,
        observed_dim=100,
        n_states=2,
        state_timing='equipartition',
        test_state=-1,
        seed=11,
        save=f'{results}/corr_expt-stateident/corr_expt-stateident.svg',
    )
    """

    print('\n-----------------------------------------')
    print('Experiment 2: Single-Subject Parcellation')
    print('-----------------------------------------')
    os.makedirs(f'{results}/corr_expt-parcellation1', exist_ok=True)
    unsupervised_state_detection_experiment(
        lr=0.01,
        max_epoch=1001,
        log_interval=20,
        smoothness_nu=5,
        symbimodal_nu=1,
        entropy_nu=0.01,
        equilibrium_nu=5000,
        dist_nu=1,
        time_dim=1000,
        subject_dim=1,
        latent_dim=10,
        observed_dim=30,
        state_weight=1,
        subject_weight=1,
        n_states=6,
        seed=1,
        save=f'{results}/corr_expt-parcellation1/corr_expt-parcellation1',
    )
