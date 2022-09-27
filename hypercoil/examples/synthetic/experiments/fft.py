#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Frequency product experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simple experiments using the frequency product module.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import distrax
import optax
from functools import partial
from typing import Callable, List, Optional
from matplotlib.pyplot import close
from hypercoil.engine.noise import (
    ScalarIIDMulStochasticTransform,
    StochasticParameter,
)
from hypercoil.engine.paramutil import PyTree, Tensor, _to_jax_array
from hypercoil.functional import corr, complex_decompose, linear_distance, sym2vec
from hypercoil.init import FreqFilterSpec
from hypercoil.init.base import DistributionInitialiser
from hypercoil.init.mapparam import ProbabilitySimplexParameter
from hypercoil.nn import (
    FrequencyDomainFilter,
    UnaryCovarianceUW,
)
from hypercoil.loss.scalarise import (
    mean_scalarise,
    meansq_scalarise,
    vnorm_scalarise,
    max_scalarise
)
from hypercoil.loss.scheme import (
    LossScheme,
    LossApply,
    LossArgument,
)
from hypercoil.loss.nn import (
    DispersionLoss,
    MSELoss,
    NormedLoss,
    EntropyLoss,
    MultivariateKurtosis,
    SmoothnessLoss,
    BimodalSymmetricLoss,
)
from hypercoil.examples.synthetic.scripts.filter import (
    synthesise_across_bands,
    plot_frequency_partition,
    plot_mvkurtosis
)
from hypercoil.examples.synthetic.scripts.mix import synthesise_mixture
from hypercoil.examples.synthetic.experiments.overfit_plot import (
    overfit_and_plot_progress
)


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
    ampl, phase = complex_decompose(_to_jax_array(model.filter.weight))
    return ampl


def frequency_band_identification_experiment(
    lr: float = 5e-3,
    max_epoch: int = 300,
    latent_dim: int = 7,
    observed_dim: int = 100,
    time_dim: int = 1000,
    smoothness_nu: float = 0.2,
    symbimodal_nu: float = 0.05,
    dispersion_nu: float = 0.05,
    l2_nu: float = 0.015,
    entropy_nu: float = 0.1,
    test_band: int = 0,
    log_interval: int = 5,
    supervised: bool = True,
    save: Optional[str] = None,
    *,
    key: int,
):
    key = jax.random.PRNGKey(key)
    key_d, key_n, key_m, key_l = jax.random.split(key, 4)
    X, Y, target = synthesise_across_bands(
        bands=DEFAULT_BANDS,
        latent_dim=latent_dim,
        observed_dim=observed_dim,
        key=key_d
    )
    max_tol_score = jnp.sqrt(.01 * time_dim)
    freq_dim = time_dim // 2 + 1

    survival_prob=0.2
    dropout = ScalarIIDMulStochasticTransform(
        distribution=distrax.Bernoulli(probs=survival_prob),
        inference=False,
        key=key_n
    )

    if supervised:
        target = FreqFilterSpec(Wn=target[test_band], ftype='ideal')
        fftfilter = FrequencyDomainFilter(
            freq_dim=freq_dim,
            num_channels=1,
            key=key_m
        )
        # Without this initialisation, our gradient catastrophically explodes!
        # No idea why this is the case, but it's a good reminder that
        # initialisation is important! (Thanks copilot!)
        #
        # I've gone ahead and made this (0 phase) the default initialisation
        # for FrequencyDomainFilter. There's a good chance it's not always a
        # good idea, but it's a decent starting point.
        fftfilter = DistributionInitialiser.init(
            fftfilter, distribution=distrax.Normal(0.5, 0.01), key=key_m,
        )
    else:
        target = [FreqFilterSpec(
            Wn=target[i], ftype='ideal')
            for i in range(N_BANDS)]
        fftfilter = FrequencyDomainFilter(
            freq_dim=freq_dim,
            num_channels=(N_BANDS + 1),
            key=key_m
        )
        fftfilter = DistributionInitialiser.init(
            fftfilter, distribution=distrax.Normal(0.5, 0.01), key=key_m,
        )
        fftfilter = ProbabilitySimplexParameter.map(
            fftfilter, axis=0, where='weight')

    class FFTSeq(eqx.Module):
        filter: PyTree
        cov: PyTree
        weight: Optional[Tensor] = None

        def __call__(
            self,
            X: Tensor,
            *,
            key: Optional['jax.random.PRNGKey'] = None,
        ) -> Tensor:
            key_f, key_c = jax.random.split(key)
            Y = self.filter(X, key=key_f)
            weight = self.weight
            if weight is not None:
                weight = jax.lax.stop_gradient(_to_jax_array(weight))
            return self.cov(Y, weight=weight, key=key_c)

    model = FFTSeq(
        filter=fftfilter,
        cov=UnaryCovarianceUW(
            estimator=corr,
            dim=time_dim),
        weight=jnp.ones((1, time_dim)),
    )
    model = StochasticParameter.wrap(
        model, where='weight', transform=dropout)

    Y = Y[test_band]

    # SGD tends to get stuck in worse minima here
    #opt = optax.sgd(learning_rate=lr)
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    if supervised:
        loss = MSELoss()
        reg = LossScheme([
            SmoothnessLoss(
                name='Smoothness',
                nu=smoothness_nu,
                scalarisation=mean_scalarise(inner=vnorm_scalarise()),),
            BimodalSymmetricLoss(
                name='BimodalSymmetric',
                nu=symbimodal_nu,
                scalarisation=mean_scalarise(inner=vnorm_scalarise()),),
            NormedLoss(
                name='L2',
                nu=l2_nu,
                axis=-1,), # The axis shouldn't be set here, but we don't want to
                           # have to figure out the hyperparameters again.
        ], apply=lambda model: amplitude(model))
        spectrum = target.initialise_spectrum(worN=model.filter.dim, key=key_m)
        target = spectrum.T

        #TODO: investigate further --
        # Phase reg doesn't seem to work . . .
        # We get phase randomisation where the amplitude is close to zero
        # and placing a too strict penalty on phase for some reason messes
        # up the amplitude structure

        overfit_and_plot_progress(
            out_fig=save, model=model, optim_state=opt_state, optim=opt,
            reg=reg, loss=loss, max_epoch=max_epoch,
            X=X, Y=Y, target=target,
            log_interval=log_interval, plot=amplitude, key=key_l,
        )

        ampl, _ = complex_decompose(model.filter.weight)
        target = target.squeeze()
        solution = ampl.squeeze()
        score = loss(target, solution)
        # This is the score if every guess were exactly 0.1 from the target.
        # (already far better than random chance)
        assert(score < max_tol_score)
    else:
        loss = LossScheme([
            LossScheme([
                SmoothnessLoss(
                    name='Smoothness',
                    nu=smoothness_nu,
                    scalarisation=mean_scalarise(
                        inner=vnorm_scalarise(p=2, axis=-1)
                    ),
                    axis=-1),
                EntropyLoss(
                    name='Entropy',
                    nu=entropy_nu,
                    axis=0)
            ],
            apply = lambda arg: amplitude(arg.model)),
            LossScheme([
                BimodalSymmetricLoss(
                    name='BimodalSymmetric',
                    nu=symbimodal_nu,
                    scalarisation=meansq_scalarise(),
                    modes=(-1, 1)),
                DispersionLoss(
                    name='Dispersion',
                    nu=dispersion_nu,
                    scalarisation=mean_scalarise(inner=max_scalarise(axis=-1)),
                    metric=linear_distance)
            ], apply = lambda arg: sym2vec(arg.corr)),
        ])
        spectrum = [t.initialise_spectrum(worN=model.filter.dim, key=key_m)
                    for t in target]
        target = [s.T for s in spectrum]
        frequency_band_partition_experiment(
            model=model,
            opt=opt,
            opt_state=opt_state,
            loss=loss,
            max_epoch=max_epoch,
            X=X,
            target=target,
            log_interval=log_interval,
            save=save,
            key=key_l,
        )


def frequency_band_partition_experiment(
    model: PyTree,
    opt: 'optax.GradientTransformation',
    opt_state: PyTree,
    loss: Callable,
    max_epoch: int,
    X: Tensor,
    target: List[Tensor],
    log_interval: int,
    save: Optional[str] = None,
    *,
    key: 'jax.random.PRNGKey',
):
    def forward(
        model: PyTree,
        X: Tensor,
        loss: Callable,
        key: 'jax.random.PRNGKey'
    ):
        key_m, key_l = jax.random.split(key)
        corr = model(X, key=key_m)[:-1]
        arg = LossArgument(model=model, X=X, corr=corr)
        loss_epoch, meta = loss(arg, key=key_l)
        return loss_epoch, meta

    losses = []
    for epoch in range(max_epoch):
        key = jax.random.split(key, 1)[0]
        (loss_epoch, meta), grad = eqx.filter_jit(eqx.filter_value_and_grad(
            forward, has_aux=True
        ))(model, X, loss, key=key)
        # for k, v in meta.items():
        #     print(f'{k}: {v.value:.4f}')
        losses += [loss_epoch.item()]
        updates, opt_state = opt.update(
            eqx.filter(grad, eqx.is_inexact_array),
            opt_state,
            eqx.filter(model, eqx.is_inexact_array),
        )
        model = eqx.apply_updates(model, updates)
        if epoch % log_interval == 0:
            print(f'[ Epoch {epoch} | Loss {loss_epoch} ]')
            plot_frequency_partition(
                bands=DEFAULT_BANDS,
                filter=model.filter,
                save=f'{save}-{epoch:08}.png'
            )
            close('all')


def dynamic_band_identification_experiment(
    lr: float = 1e-3,
    max_epoch: int = 2501,
    latent_dim: int = 10,
    observed_dim: int = 50,
    time_dim: int = 1000,
    n_switches: int = 20,
    mvkurtosis_nu: float = 0.001,
    smoothness_nu: float = 0.2,
    symbimodal_nu: float = 0.05,
    l2_nu: float = 0.25,
    mvkurtosis_l2: float = 0.01,
    log_interval: int = 100,
    save: Optional[str] = None,
    *,
    key: int,
):
    key = jax.random.PRNGKey(key)
    key_b, key_s, key_m, key_l = jax.random.split(key, 4)
    assert (time_dim / n_switches == time_dim // n_switches)

    def amplitude(weight):
        """
        Accessory function for use by regularisers, so that they can operate
        specifically on the amplitude of the filter's response curve.
        """
        # Or, we could just use the absolute value . . .
        ampl, phase = complex_decompose(_to_jax_array(weight))
        return ampl

    bands = [(0.1, 0.2), (0.2, 0.3), (0.4, 0.5), (0.5, 0.6)]
    keys = jax.random.split(key_b, len(bands))
    bands = [synthesise_mixture(
        observed_dim=observed_dim,
        latent_dim=latent_dim,
        time_dim=time_dim,
        lp=lp,
        hp=hp,
        key=k,
    ) for k, (hp, lp) in zip(keys, bands)]

    keys = jax.random.split(key_s, n_switches)
    dynamic_band = [synthesise_mixture(
        observed_dim=observed_dim,
        latent_dim=latent_dim,
        time_dim=(time_dim // n_switches),
        lp=0.4,
        hp=0.3,
        key=k,
    ) for k in keys]
    dynamic_band = jnp.concatenate(dynamic_band, -1)

    signal = sum(bands) + dynamic_band
    signal = signal - signal.mean(-1, keepdims=True)
    signal = signal / signal.std(-1, keepdims=True)

    freq_dim = time_dim // 2 + 1
    fftfilter = FrequencyDomainFilter(
        freq_dim=freq_dim,
        num_channels=1,
        key=key_m,
    )

    loss = LossScheme([
        LossApply(
            MultivariateKurtosis(
                name='MultivariateKurtosis',
                nu=mvkurtosis_nu,
                l2=mvkurtosis_l2,),
            apply=lambda arg: arg.ts_filtered
        ),
        LossScheme([
            SmoothnessLoss(
                name='Smoothness',
                nu=smoothness_nu,
                scalarisation=mean_scalarise(inner=vnorm_scalarise()),),
            BimodalSymmetricLoss(
                name='BimodalSymmetric',
                nu=symbimodal_nu,
                scalarisation=mean_scalarise(inner=vnorm_scalarise()),),
            NormedLoss(
                name='L2',
                nu=l2_nu)
        ], apply=lambda arg: amplitude(arg.weight))
    ])

    X = signal
    opt = optax.adam(learning_rate=lr)
    opt_state = opt.init(eqx.filter(fftfilter, eqx.is_inexact_array))

    def forward(model, X, loss, key):
        key_m, key_l = jax.random.split(key)
        ts_filtered = model(X, key=key_m)
        arg = LossArgument(
            ts_filtered=ts_filtered,
            weight=model.weight
        )
        loss_epoch, meta = loss(arg, key=key_l)
        return loss_epoch, meta


    losses = []
    for epoch in range(max_epoch):
        key_l = jax.random.split(key_l, 1)[0]
        (loss_epoch, meta), grad = eqx.filter_jit(eqx.filter_value_and_grad(
            forward, has_aux=True
        ))(fftfilter, X, loss, key=key)
        # for k, v in meta.items():
        #     print(f'{k}: {v.value:.4f}')
        losses += [loss_epoch.item()]
        updates, opt_state = opt.update(
            eqx.filter(grad, eqx.is_inexact_array),
            opt_state,
            eqx.filter(fftfilter, eqx.is_inexact_array),
        )
        fftfilter = eqx.apply_updates(fftfilter, updates)
        if epoch % log_interval == 0:
            print(f'[ Epoch {epoch} | Total loss : {losses[-1]} ]')

    all_bands = [
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 0.6)
    ]
    test_filter = FrequencyDomainFilter(
        freq_dim=freq_dim,
        num_channels=1,
        key=key,
    )
    plot_mvkurtosis(
        fftfilter=test_filter,
        weight=_to_jax_array(fftfilter.weight),
        input=X,
        bands=all_bands,
        nu=mvkurtosis_nu,
        l2=mvkurtosis_l2,
        save=save
    )


def main():
    from hypercoil.examples.synthetic.experiments.run import (
        run_layer_experiments
    )
    run_layer_experiments('fft')


if __name__ == '__main__':
    main()
