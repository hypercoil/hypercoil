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
from matplotlib.pyplot import close
from hypercoil.init import FreqFilterSpec
from hypercoil.functional import corr
from hypercoil.functional.activation import complex_decompose
from hypercoil.functional.domain import (
    Identity,
    AmplitudeMultiLogit
)
from hypercoil.functional.noise import UnstructuredDropoutSource
from hypercoil.nn import (
    FrequencyDomainFilter,
    UnaryCovarianceUW
)
from hypercoil.reg import (
    RegularisationScheme,
    Entropy,
    NormedRegularisation,
    SmoothnessPenalty,
    SymmetricBimodal
)
from hypercoil.synth.filter import (
    synthesise_across_bands,
    plot_frequency_partition
)
from hypercoil.synth.experiments.overfit_plot import overfit_and_plot_progress


DEFAULT_BANDS = (
    (0.05, 0.1),
    (0.1, 0.3),
    (0.3, 0.6)
)


N_BANDS = len(DEFAULT_BANDS)


def amplitude(model, X=None, Y=None):
    """
    Accessory functions for use by regularisers, so that they can operate
    specifically on the amplitude of the filter's response curve.
    """
    ampl, phase = complex_decompose(model[0].weight)
    return ampl


def correlations(model, X, Y):
    """
    Accessory functions for use by regularisers, so that they can operate
    specifically on the model output correlation matrices.
    """
    return Y


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
    entropy_nu=0.1,
    test_band=0,
    log_interval=5,
    supervised=True,
    save=None
):
    if seed is not None: torch.manual_seed(seed)
    np.random.seed(seed)
    X, Y, target = synthesise_across_bands(
        bands=DEFAULT_BANDS,
        latent_dim=latent_dim,
        observed_dim=observed_dim,
        seed=seed
    )
    max_tol_score = np.sqrt(.01 * time_dim)
    freq_dim = time_dim // 2 + 1

    survival_prob=0.2
    drop = UnstructuredDropoutSource(
        distr=torch.distributions.Bernoulli(survival_prob),
        training=True
    )

    if supervised:
        target = FreqFilterSpec(Wn=target[test_band], ftype='ideal')
        filter_specs = [FreqFilterSpec(
            Wn=None, ftype='randn', btype=None, # clamps = [{0: 0}]
            ampl_scale=0.01, phase_scale=0.01
        )]
        fftfilter = FrequencyDomainFilter(
            dim=freq_dim,
            filter_specs=filter_specs,
            domain=Identity()
        )
    else:
        target = [FreqFilterSpec(
            Wn=target[i], ftype='ideal')
            for i in range(N_BANDS)]
        filter_specs = [FreqFilterSpec(
            Wn=None, ftype='randn', btype=None, # clamps = [{0: 0}]
            ampl_scale=0.01, phase_scale=0.01
        ) for _ in range(N_BANDS + 1)]
        fftfilter = FrequencyDomainFilter(
            dim=freq_dim,
            filter_specs=filter_specs,
            domain=AmplitudeMultiLogit(axis=0)
        )

    model = torch.nn.Sequential(
        fftfilter,
        UnaryCovarianceUW(
            dim=time_dim,
            estimator=corr,
            dropout=drop
        )
    )

    X = torch.Tensor(X)
    Y = torch.Tensor(Y[test_band])

    # SGD tends to get stuck in worse minima here
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss = torch.nn.MSELoss()
    if supervised:
        reg_ampl = RegularisationScheme([
            SmoothnessPenalty(nu=smoothness_nu),
            SymmetricBimodal(nu=symbimodal_nu),
            NormedRegularisation(nu=l2_nu)
        ])
        target.initialise_spectrum(
            model[0].dim, domain=Identity())
        target = target.spectrum.T
    else:
        reg_ampl = RegularisationScheme([
            SmoothnessPenalty(nu=smoothness_nu),
            Entropy(nu=entropy_nu)
        ])
        reg_corr = SymmetricBimodal(nu=symbimodal_nu, modes=(-1, 1))
        reg = [
            (reg_ampl, amplitude),
            (reg_corr, correlations)
        ]
        for t in target:
            t.initialise_spectrum(model[0].dim, domain=Identity())
        target = [t.spectrum.T for t in target]
        frequency_band_partition_experiment(
            model=model,
            opt=opt,
            reg=reg,
            loss=loss,
            max_epoch=max_epoch,
            X=X,
            target=target,
            log_interval=log_interval,
            save=save
        )
        return
    #TODO: investigate further --
    # Phase reg doesn't seem to work . . .
    # We get phase randomisation where the amplitude is close to zero
    # and placing a too strict penalty on phase for some reason messes
    # up the amplitude structure

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


def frequency_band_partition_experiment(
    model, opt, reg, loss, max_epoch, X, target, log_interval, save
):
    losses = []
    for epoch in range(max_epoch):
        corr = model(X)[:-1]
        #print(corr.shape)
        loss_epoch = 0
        for r, penalise in reg:
            loss_epoch += r(penalise(model, X, corr))
        loss_epoch.backward()
        losses += [loss_epoch.detach().item()]
        opt.step()
        model.zero_grad()
        if epoch % log_interval == 0:
            print(f'[ Epoch {epoch} | Loss {loss_epoch} ]')
            plot_frequency_partition(
                bands=DEFAULT_BANDS,
                filter=model[0],
                save=f'{save}-{epoch:08}.png'
            )
            close('all')

        """
        if epoch == 0:
            print('Statistics at train start:')
            print(f'- Compactness: {reg[0](mnorm)}')
            print(f'- Entropy: {reg[1](mnorm)}')
            print(f'- Equilibrium: {reg[2](mnorm)}')
            print(f'- Second Moment: {reg2m(mnorm, X)}')
            print(f'- Log det: {regdet(ts_parc)}')
        loss_epoch = 0
        if supervised:
            loss_epoch = loss(corrmat[uniq_idx], Y[uniq_idx])
        loss_epoch = loss_epoch + reg(mnorm) + reg2m(mnorm, X) + regdet(ts_parc)
        loss_epoch.backward()
        losses += [loss_epoch.detach().item()]
        opt.step()
        model.grad.zero_()
        if epoch % log_interval == 0:
            print(f'[ Epoch {epoch} | Loss {loss_epoch} ]')
            plot_atlas(
                mnorm,
                d=image_dim,
                c=parcel_count,
                saveh=f'{save}hard-{epoch:08}.png',
                saves=f'{save}soft-{epoch:08}.png'
            )
            close('all')
        """


if __name__ == '__main__':
    import os, hypercoil
    results = os.path.abspath(f'{hypercoil.__file__}/../results')

    print('\n-----------------------------------------')
    print('Experiment 1: Band 1 Identification')
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
    print('Experiment 2: Band 2 Identification')
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
    print('Experiment 3: Band 3 Identification')
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

    print('\n-----------------------------------------')
    print('Experiment 4: Band Parcellation')
    print('-----------------------------------------')
    os.makedirs(f'{results}/fft_expt-parcellation', exist_ok=True)
    frequency_band_identification_experiment(
        lr=5e-3,
        seed=0,
        max_epoch=600,
        latent_dim=7,
        observed_dim=100,
        time_dim=1000,
        smoothness_nu=0.4,
        symbimodal_nu=0.05,
        entropy_nu=0.1,
        l2_nu=0.015,
        test_band=0,
        log_interval=10,
        supervised=False,
        save=f'{results}/fft_expt-parcellation/fft_expt-parcellation.png',
    )
