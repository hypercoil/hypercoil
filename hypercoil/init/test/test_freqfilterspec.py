# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for IIR filter initialisation
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
from pkg_resources import resource_filename as pkgrf
from hypercoil.functional.utils import Tensor
from hypercoil.init.mapparam import AmplitudeTanhMappedParameter
from hypercoil.init.freqfilter import (
    FreqFilterSpec,
    freqfilter_init,
    clamp_init,
    FreqFilterInitialiser,
)


class TestFreqFilter:

    def plot_ampl_response(self, spectra, names, path):
        fig = plt.figure(figsize=(10, 10))
        for spectrum in spectra:
            freqs = np.linspace(0, 1, spectrum.shape[-1])
            spectrum = np.abs(spectrum)
            plt.plot(freqs, spectrum)
        plt.legend(names)
        results = pkgrf(
            'hypercoil',
            'results/'
        )
        plt.suptitle('Emulated amplitude response')
        plt.title('Do not use this as a guide: it is just a test', fontsize=8)
        fig.savefig(f'{results}/freqfilter_{path}.png', bbox_inches='tight')

    def test_spectra(self):
        N = (1, 4)
        Wn = ((0.1, 0.3), (0.4, 0.6))
        filter_specs = (
            FreqFilterSpec(Wn=[0.1, 0.3], ftype='butter'),
            FreqFilterSpec(Wn=Wn, N=N, ftype='butter'),
            FreqFilterSpec(Wn=Wn, ftype='ideal'),
            FreqFilterSpec(Wn=[0.1, 0.2], N=[2, 2], btype='lowpass'),
            FreqFilterSpec(Wn=Wn, N=N, ftype='cheby1', rp=0.01),
            FreqFilterSpec(Wn=Wn, N=N, ftype='cheby2', rs=20),
            FreqFilterSpec(Wn=Wn, N=N, ftype='ellip', rs=20, rp=0.1),
            FreqFilterSpec(Wn=((0.2, 0.3), (0.4, 0.6)), N=N,
                           ftype='bessel', norm='amplitude'),
            FreqFilterSpec(Wn=Wn, ftype='randn'),
        )
        filter_names = (
            'butter', 'ideal', 'cheby1', 'cheby2', 'ellip', 'bessel', 'randn'
        )
        filter_idx = (1, 2, 4, 5, 6, 7, 8)

        spectra = [
            s.initialise_spectrum(key=jax.random.PRNGKey(0), worN=200)
            for s in filter_specs
        ]
        spectra = [s[0] for i, s in enumerate(spectra) if i in filter_idx]
        self.plot_ampl_response(spectra, filter_names, 'spectra')
        out = freqfilter_init(
            shape=(3, 1, 200),
            filter_specs=filter_specs,
            # Key is split for consistency with the call inside the
            # initialiser
            key=jax.random.split(jax.random.PRNGKey(0), 1)[0])
        n_filters = sum([len(s.Wn) for s in filter_specs])
        assert out.shape == (3, n_filters, 200)

        model = eqx.nn.Linear(
            in_features=200, out_features=1,
            key=jax.random.PRNGKey(0))
        model = FreqFilterInitialiser.init(
            model,
            filter_specs=filter_specs,
            key=jax.random.PRNGKey(0)
        )
        weight_out = model.weight
        assert weight_out.shape == (n_filters, 200)
        assert np.allclose(weight_out, out[0])

    def test_clamped_spectra(self):
        N = (1, 4)
        Wn = ((0.1, 0.3), (0.4, 0.6))
        clamped_specs = [
            FreqFilterSpec(Wn=[0.1, 0.3]),
            FreqFilterSpec(Wn=Wn, clamps=[{0.1: 1}]), # broadcast clamp
            FreqFilterSpec(Wn=[0.1, 0.3], clamps=[{0.1: 0, 0.5:1}]),
            FreqFilterSpec(Wn=Wn, N=N, clamps=[{0.05: 1, 0.1: 0},
                                               {0.2: 0, 0.5: 1}])
        ]
        points, values = clamp_init(
            shape=(3, 1, 200),
            filter_specs=clamped_specs,
            key=jax.random.PRNGKey(0))
        n_filters = sum([len(s.Wn) for s in clamped_specs])
        assert points.shape == values.shape
        assert points.shape == (n_filters, 200)
        assert points.sum() == 8
        assert values.sum() == 5
        assert np.logical_or(values == 0, values == 1).all()

        #TODO: replace this with an actual freqfilter module after it's
        #      translated to jax
        class FreqFilter(eqx.Module):
            weight: Tensor
            clamp: tuple[Tensor, Tensor]
            def __init__(self, worN, key):
                self.weight = jnp.zeros((1, worN))
                self.clamp = (
                    jnp.zeros((1, worN)),
                    jnp.zeros((1, worN))
                )
        model = FreqFilter(worN=200, key=jax.random.PRNGKey(0))
        model = FreqFilterInitialiser.init(
            model,
            filter_specs=clamped_specs,
            clamp_name='clamp',
            key=jax.random.PRNGKey(0)
        )
        points_out, values_out = model.clamp
        assert points_out.shape == values_out.shape
        assert points_out.shape == (n_filters, 200)
        assert (points_out == points).all()
        assert np.allclose(values_out, values)
