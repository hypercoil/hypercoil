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
from hypercoil.functional.cov import corr
from hypercoil.functional.domain import Identity
from hypercoil.init.base import (
    DistributionInitialiser
)
from hypercoil.nn.cov import UnaryCovariance
from hypercoil.reg import (
    RegularisationScheme,
    SmoothnessPenalty,
    SymmetricBimodal
)
from hypercoil.synth.corr import synthesise_state_transition
from hypercoil.synth.experiments.overfit_plot import overfit_and_plot_progress


def state_detection_experiment(
    lr=1,
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


if __name__ == '__main__':
    import os, hypercoil
    results = os.path.abspath(f'{hypercoil.__file__}/../results')

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
