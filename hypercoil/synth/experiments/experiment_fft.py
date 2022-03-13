#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Frequency product experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simple experiments using the frequency product module.
"""
import torch
import numpy as np
from hypercoil.init import FreqFilterSpec
from hypercoil.functional import corr
from hypercoil.functional.activation import complex_decompose
from hypercoil.functional.domain import Identity
from hypercoil.functional.noise import UnstructuredDropoutSource
from hypercoil.nn import (
    FrequencyDomainFilter,
    UnaryCovarianceUW
)
from hypercoil.reg import (
    RegularisationScheme,
    NormedRegularisation,
    SmoothnessPenalty,
    SymmetricBimodal
)
from hypercoil.synth.filter import synthesise_across_bands
from hypercoil.synth.experiments.overfit_plot import overfit_and_plot_progress


def amplitude(model):
    """
    Accessory functions for use by regularisers, so that they can operate
    specifically on the amplitude of the filter's response curve.
    """
    ampl, phase = complex_decompose(model[0].weight)
    return ampl


def frequency_band_identification_experiment(
    lr=5e-3,
    seed=None,
    max_epoch=300,
    latent_dim=7,
    observed_dim=100,
    time_dim=1000,
    smoothness_nu=0.2,
    symbimodal_nu=0.05,
    l2_nu=0.015,
    test_band=0,
    log_interval=5,
    save=None
):
    if seed is not None: torch.manual_seed(seed)
    np.random.seed(seed)
    filter_specs = [FreqFilterSpec(
        Wn=None, ftype='randn', btype=None, # clamps = [{0: 0}]
        ampl_scale=0.01, phase_scale=0.01
    )]
    X, Y, target = synthesise_across_bands(
        latent_dim=latent_dim,
        observed_dim=observed_dim,
        seed=seed
    )
    target = FreqFilterSpec(Wn=target[test_band], ftype='ideal')
    max_tol_score = np.sqrt(.01 * time_dim)
    freq_dim = time_dim // 2 + 1

    survival_prob=0.2
    drop = UnstructuredDropoutSource(
        distr=torch.distributions.Bernoulli(survival_prob),
        training=True
    )

    model = torch.nn.Sequential(
        FrequencyDomainFilter(
            dim=freq_dim,
            filter_specs=filter_specs,
            domain=Identity()
        ),
        UnaryCovarianceUW(
            dim=time_dim,
            estimator=corr,
            dropout=drop
        )
    )

    # SGD tends to get stuck in worse minima here
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.MSELoss()
    reg_ampl = RegularisationScheme([
        SmoothnessPenalty(nu=smoothness_nu),
        SymmetricBimodal(nu=symbimodal_nu),
        NormedRegularisation(nu=l2_nu)
    ])
    #TODO: investigate further --
    # Phase reg doesn't seem to work . . .
    # We get phase randomisation where the amplitude is close to zero
    # and placing a too strict penalty on phase for some reason messes
    # up the amplitude structure

    X = torch.Tensor(X)
    Y = torch.Tensor(Y[test_band])
    target.initialise_spectrum(
        model[0].dim, domain=Identity())
    target = target.spectrum.T

    overfit_and_plot_progress(
        out_fig=save, model=model, optim=opt, reg=reg_ampl, loss=loss,
        max_epoch=max_epoch, X=X, Y=Y, target=target, seed=seed,
        log_interval=log_interval, penalise=amplitude, plot=amplitude
    )

    ampl, _ = complex_decompose(model[0].weight)
    target = torch.Tensor(target).squeeze()
    solution = ampl.squeeze()
    score = loss(target, solution)
    # This is the score if every guess were exactly 0.1 from the target.
    # (already far better than random chance)
    assert(score < max_tol_score)


if __name__ == '__main__':
    import os, hypercoil
    results = os.path.abspath(f'{hypercoil.__file__}/../results')

    print('\n-----------------------------------------')
    print('Experiment 1: Band Identification')
    print('-----------------------------------------')
    os.makedirs(f'{results}/fft_expt-bandident', exist_ok=True)
    frequency_band_identification_experiment(
        lr=5e-3,
        seed=0,
        max_epoch=300,
        latent_dim=7,
        observed_dim=100,
        time_dim=1000,
        smoothness_nu=0.2,
        symbimodal_nu=0.05,
        l2_nu=0.015,
        test_band=0,
        log_interval=5,
        save=f'{results}/fft_expt-bandident/fft_expt-bandident0.svg',
    )

    print('\n-----------------------------------------')
    print('Experiment 1: Band Identification')
    print('-----------------------------------------')
    os.makedirs(f'{results}/fft_expt-bandident', exist_ok=True)
    frequency_band_identification_experiment(
        lr=5e-3,
        seed=0,
        max_epoch=300,
        latent_dim=7,
        observed_dim=100,
        time_dim=1000,
        smoothness_nu=0.2,
        symbimodal_nu=0.05,
        l2_nu=0.015,
        test_band=1,
        log_interval=5,
        save=f'{results}/fft_expt-bandident/fft_expt-bandident1.svg',
    )

    print('\n-----------------------------------------')
    print('Experiment 1: Band Identification')
    print('-----------------------------------------')
    os.makedirs(f'{results}/fft_expt-bandident', exist_ok=True)
    frequency_band_identification_experiment(
        lr=5e-3,
        seed=0,
        max_epoch=300,
        latent_dim=7,
        observed_dim=100,
        time_dim=1000,
        smoothness_nu=0.2,
        symbimodal_nu=0.05,
        l2_nu=0.015,
        test_band=2,
        log_interval=5,
        save=f'{results}/fft_expt-bandident/fft_expt-bandident2.svg',
    )

