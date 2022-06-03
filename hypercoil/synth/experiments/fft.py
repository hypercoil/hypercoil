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
from hypercoil.functional import corr, complex_decompose
from hypercoil.init.domain import (
    Identity,
    AmplitudeMultiLogit
)
from hypercoil.functional.noise import UnstructuredDropoutSource
from hypercoil.nn import (
    FrequencyDomainFilter,
    UnaryCovarianceUW
)
from hypercoil.loss import (
    LossScheme,
    LossApply,
    LossArgument,
    Entropy,
    MultivariateKurtosis,
    NormedLoss,
    SmoothnessPenalty,
    SymmetricBimodalNorm
)
from hypercoil.synth.filter import (
    synthesise_across_bands,
    plot_frequency_partition,
    plot_mvkurtosis
)
from hypercoil.synth.mix import synthesise_mixture
from hypercoil.synth.experiments.overfit_plot import overfit_and_plot_progress


DEFAULT_BANDS = (
    (0.05, 0.1),
    (0.1, 0.3),
    (0.3, 0.6)
)


N_BANDS = len(DEFAULT_BANDS)


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
        #TODO: When the ANOML domain is working, we should use that here
        # instead.
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
    if supervised:
        loss = torch.nn.MSELoss()
        reg = LossScheme([
            SmoothnessPenalty(nu=smoothness_nu),
            SymmetricBimodalNorm(nu=symbimodal_nu),
            NormedLoss(nu=l2_nu)
        ], apply=lambda model: amplitude(model))
        target.initialise_spectrum(
            model[0].dim, domain=Identity())
        target = target.spectrum.T

        #TODO: investigate further --
        # Phase reg doesn't seem to work . . .
        # We get phase randomisation where the amplitude is close to zero
        # and placing a too strict penalty on phase for some reason messes
        # up the amplitude structure

        overfit_and_plot_progress(
            out_fig=save, model=model, optim=opt, reg=reg, loss=loss,
            max_epoch=max_epoch, X=X, Y=Y, target=target, seed=seed,
            log_interval=log_interval, plot=amplitude
        )

        ampl, _ = complex_decompose(model[0].weight)
        target = torch.Tensor(target).squeeze()
        solution = ampl.squeeze()
        score = loss(target, solution)
        # This is the score if every guess were exactly 0.1 from the target.
        # (already far better than random chance)
        assert(score < max_tol_score)
    else:
        loss = LossScheme([
            LossApply(
                loss=LossScheme([
                    SmoothnessPenalty(nu=smoothness_nu),
                    Entropy(nu=entropy_nu)
                ]),
                apply = lambda arg: amplitude(arg.model)
            ),
            LossApply(
                loss=SymmetricBimodalNorm(nu=symbimodal_nu, modes=(-1, 1)),
                apply = lambda arg: arg.corr
            )
        ])
        for t in target:
            t.initialise_spectrum(model[0].dim, domain=Identity())
        target = [t.spectrum.T for t in target]
        frequency_band_partition_experiment(
            model=model,
            opt=opt,
            loss=loss,
            max_epoch=max_epoch,
            X=X,
            target=target,
            log_interval=log_interval,
            save=save
        )
        return


def frequency_band_partition_experiment(
    model, opt, loss, max_epoch, X, target, log_interval, save
):
    losses = []
    for epoch in range(max_epoch):
        corr = model(X)[:-1]
        arg = LossArgument(model=model, X=X, corr=corr)
        if epoch == 0:
            print('Statistics at train start:')
            loss(arg, verbose=True)
        loss_epoch = loss(arg)
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


def dynamic_band_identification_experiment(
    lr=1e-3,
    seed=None,
    max_epoch=2501,
    latent_dim=10,
    observed_dim=50,
    time_dim=1000,
    n_switches=20,
    mvkurtosis_nu=0.001,
    smoothness_nu=0.2,
    symbimodal_nu=0.05,
    l2_nu=0.25,
    mvkurtosis_l2=0.01,
    log_interval=100,
    save=None
):
    assert (time_dim / n_switches == time_dim // n_switches)

    def amplitude(weight):
        """
        Accessory function for use by regularisers, so that they can operate
        specifically on the amplitude of the filter's response curve.
        """
        ampl, phase = complex_decompose(weight)
        return ampl

    bands = [(0.1, 0.2), (0.2, 0.3), (0.4, 0.5), (0.5, 0.6)]
    if seed is not None:
        seeds = [seed + i for i in range(len(bands))]
    else:
        seeds = [None for _ in bands]
    seed_offset = len(bands)
    bands = [synthesise_mixture(
        observed_dim=observed_dim,
        latent_dim=latent_dim,
        time_dim=time_dim,
        lp=lp,
        hp=hp,
        seed=s
    ) for s, (hp, lp) in zip(seeds, bands)]

    if seed is not None:
        seeds = [seed + i + seed_offset for i in range(n_switches)]
    else:
        seeds = [None for _ in range(n_switches)]
    dynamic_band = [synthesise_mixture(
        observed_dim=observed_dim,
        latent_dim=latent_dim,
        time_dim=(time_dim // n_switches),
        lp=0.4,
        hp=0.3,
        seed=s
    ) for s in seeds]
    dynamic_band = np.concatenate(dynamic_band, -1)

    signal = sum(bands) + dynamic_band
    signal = signal - signal.mean(-1, keepdims=True)
    signal = signal / signal.std(-1, keepdims=True)

    freq_dim = time_dim // 2 + 1
    filter_specs = [FreqFilterSpec(
        Wn=None, ftype='randn', btype=None,
        ampl_scale=0.01, phase_scale=0.01
    )]
    fftfilter = FrequencyDomainFilter(
        dim=freq_dim,
        filter_specs=filter_specs,
        domain=Identity()
    )

    loss = LossScheme([
        LossApply(
            MultivariateKurtosis(nu=mvkurtosis_nu, l2=mvkurtosis_l2),
            apply=lambda arg: arg.ts_filtered
        ),
        LossScheme([
            SmoothnessPenalty(nu=smoothness_nu),
            SymmetricBimodalNorm(nu=symbimodal_nu),
            NormedLoss(nu=l2_nu)
        ], apply=lambda arg: amplitude(arg.weight))
    ])

    X = torch.tensor(signal, dtype=torch.float)
    opt = torch.optim.Adam(fftfilter.parameters(), lr=lr)

    losses = []
    for epoch in range(max_epoch):
        ts_filtered = fftfilter(X)
        arg = LossArgument(
            ts_filtered=ts_filtered,
            weight=fftfilter.weight
        )
        loss_epoch = loss(arg, verbose=(epoch % log_interval == 0))
        loss_epoch.backward()
        losses += [loss_epoch.detach().item()]
        if epoch % log_interval == 0:
            print(f'[ Epoch {epoch} | Total loss : {losses[-1]} ]')
        opt.step()
        fftfilter.weight.grad.zero_()

    all_bands = [
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 0.6)
    ]
    test_filter = FrequencyDomainFilter(
        dim=freq_dim,
        filter_specs=filter_specs,
        domain=Identity()
    )
    plot_mvkurtosis(
        fftfilter=test_filter,
        weight=fftfilter.weight.detach(),
        input=X,
        bands=all_bands,
        nu=mvkurtosis_nu,
        l2=mvkurtosis_l2,
        save=save
    )


def main():
    from hypercoil.synth.experiments.run import run_layer_experiments
    run_layer_experiments('fft')


if __name__ == '__main__':
    main()
