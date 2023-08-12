# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data synthesis
~~~~~~~~~~~~~~
Test covariance- and spectrum-matched data synthesis.
"""
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from pkg_resources import resource_filename as pkgrf
from hypercoil.functional import (
    corr, complex_decompose
)
from hypercoil.neuro.synth import (
    match_reference,
    synthesise_matched,
    synthesise_from_cov_and_spectrum
)


def amplitude(sig):
    ampl, _ = complex_decompose(sig)
    return ampl


def plot_results(ref, matched, orig=None, save=None):
    if isinstance(ref, tuple):
        ref_spec, ref_cov = ref
    else:
        ref_spec = jnp.fft.rfft(ref)
        ref_cov = corr(ref)
    plt.figure()
    plt.title('Spectra')
    if orig is not None:
        plt.plot(amplitude(jnp.fft.rfft(orig)).mean(0))
    plt.plot(amplitude(ref_spec).mean(0), color='aqua')
    plt.plot(
        amplitude(jnp.fft.rfft(matched)).mean(0),
        color='red',
        alpha=0.5
    )
    if orig is not None:
        plt.legend(['Original', 'Reference', 'Matched'])
    else:
        plt.legend(['Reference', 'Matched'])
    if save is not None:
        plt.savefig(f'{save}_spec.png')

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    if orig is not None:
        ax[0].imshow(corr(orig), cmap='RdBu_r', vmin=-0.4, vmax=0.4)
        ax[0].set_title('Original')
        ax[0].set_xticks([])
        ax[0].set_yticks([])
    ax[1].imshow(ref_cov, cmap='RdBu_r', vmin=-0.4, vmax=0.4)
    ax[1].set_title('Reference')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].imshow(corr(matched), cmap='RdBu_r', vmin=-0.4, vmax=0.4)
    ax[2].set_title('Matched')
    ax[2].set_xticks([])
    ax[2].set_yticks([])
    if save is not None:
        fig.savefig(f'{save}_cov.png')


class TestSynthesis:
    save_root = pkgrf(
        'hypercoil',
        'results/synthmatch'
    )

    def get_reference(self, obj='ts'):
        path = pkgrf(
            'hypercoil',
            'examples/synthetic/data/synth-regts/'
            f'atlas-schaefer400_desc-synth_{obj}.tsv'
        )
        data = pd.read_csv(path, sep='\t', header=None).values.T
        return jnp.array(data)

    def test_match_ts(self):
        reference = self.get_reference()
        key = jax.random.PRNGKey(0)
        synth = jax.random.normal(key=key, shape=reference.shape)
        matched = match_reference(
            signal=synth,
            reference=reference,
            use_mean=True,
        )
        plot_results(
            orig=synth,
            ref=reference,
            matched=matched,
            save=f'{self.save_root}_tsmatch',
        )

    def test_synth_ts(self):
        reference = self.get_reference()
        key = jax.random.PRNGKey(0)
        matched = synthesise_matched(
            reference=reference,
            key=key,
        )
        plot_results(
            orig=None,
            ref=reference,
            matched=matched,
            save=f'{self.save_root}_tssynth',
        )

    def test_synth_covspec(self):
        reference_spec = self.get_reference('spec').T
        reference_cov = self.get_reference('cov')
        key = jax.random.PRNGKey(0)
        matched = synthesise_from_cov_and_spectrum(
            spectrum=reference_spec,
            cov=reference_cov,
            key=key,
        )
        plot_results(
            orig=None,
            ref=(reference_spec, reference_cov),
            matched=matched,
            save=f'{self.save_root}_covspecsynth',
        )
