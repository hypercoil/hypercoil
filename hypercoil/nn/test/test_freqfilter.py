# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for frequency-domain filter layer
"""
import pytest
import numpy as np
import torch
from hypercoil.nn import FrequencyDomainFilter
from hypercoil.init.freqfilter import (
    FreqFilterSpec,
    freqfilter_init,
    clamp_init
)


import jax
import jax.numpy as jnp


class TestFreqFilter:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        N = (1, 4)
        Wn = ((0.1, 0.3), (0.4, 0.6))
        self.filter_specs = (
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
        self.clamped_specs = (
            FreqFilterSpec(Wn=[0.1, 0.3]),
            FreqFilterSpec(Wn=Wn, clamps=[{0.1: 1}]), # broadcast clamp
            FreqFilterSpec(Wn=[0.1, 0.3], clamps=[{0.1: 0, 0.5:1}]),
            FreqFilterSpec(Wn=Wn, N=N, clamps=[{0.05: 1, 0.1: 0},
                                               {0.2: 0, 0.5: 1}])
        )
        self.identity_spec = (
            FreqFilterSpec(Wn=[0, 1], ftype='ideal'),
        )

        key = jax.random.PRNGKey(0)
        self.Z1 = jax.random.uniform(key, (99,))
        self.Z2 = jax.random.uniform(key, (1, 99))
        self.Z3 = jax.random.uniform(key, (7, 99))
        self.Z4 = jax.random.uniform(key, (1, 7, 99))

        self.approx = jnp.allclose

    def test_shape_forward(self):
        key = jax.random.PRNGKey(0)
        filt = FrequencyDomainFilter.from_specs(
            self.filter_specs, freq_dim=50, key=key)
        out = filt(self.Z1)
        assert out.shape == (17, 99)
        out = filt(self.Z2)
        assert out.shape == (17, 99)
        out = filt(self.Z3)
        assert out.shape == (17, 7, 99)
        out = filt(self.Z4)
        assert out.shape == (1, 17, 7, 99)

    def test_shape_clamped_forward(self):
        key = jax.random.PRNGKey(0)
        filt = FrequencyDomainFilter.from_specs(
            self.clamped_specs, freq_dim=50, key=key)
        out = filt(self.Z1)
        assert out.shape == (6, 99)
        out = filt(self.Z2)
        assert out.shape == (6, 99)
        out = filt(self.Z3)
        assert out.shape == (6, 7, 99)
        out = filt(self.Z4)
        assert out.shape == (1, 6, 7, 99)

    def test_identity_forward(self):
        key = jax.random.PRNGKey(0)
        filt = FrequencyDomainFilter.from_specs(
            self.identity_spec, freq_dim=50, key=key)
        out = filt(self.Z1)
        assert jnp.abs(out - self.Z1).max() < 1e-5
