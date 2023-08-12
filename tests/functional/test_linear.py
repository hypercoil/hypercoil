# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for the compartmentalised linear map.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from hypercoil.functional.linear import (
    compartmentalised_linear,
    normalise_absmean, normalise_mean,
    normalise_psc, normalise_zscore
)


class TestCompartmentalisedLinear:
    def test_normalisations(self):
        x = jnp.array([-3., 1., 2., 2.])
        assert jnp.allclose(
            normalise_absmean(x, x),
            x / 8
        )
        assert jnp.allclose(
            normalise_mean(x, x),
            x / 2
        )
        assert jnp.allclose(
            normalise_psc(x, x),
            jnp.array([-700., 100., 300., 300.])
        )
        assert jnp.allclose(
            normalise_zscore(x, x),
            (x - x.mean()) / x.std()
        )

    def test_compartmentalised_linear(self):
        key = jax.random.PRNGKey(0)
        weight = {
            'A': jax.random.normal(key, (5, 100)),
            'B': jax.random.normal(key, (3, 200)),
            'C': jax.random.normal(key, (7, 300)),
        }
        limits = {
            'A': (0, 100),
            'B': (100, 200),
            'C': (200, 300),
        }
        out_idx = jax.random.choice(
            key, a=15, shape=(15,), replace=False
        ) + 1
        decoder = {
            'A': out_idx[:5],
            'B': out_idx[5:8],
            'C': out_idx[8:],
        }
        bias = jax.random.normal(key, (15,))
        data = jax.random.uniform(key, (600, 10))

        compartmentalised_linear(
            input=data,
            weight=weight,
            limits=limits,
            bias=bias,
        )

        f = eqx.filter_jit(compartmentalised_linear)
        f(
            input=data,
            weight=weight,
            bias=bias,
            limits=limits,
            decoder=decoder,
            normalisation='psc',
            forward_mode='project',
        )
