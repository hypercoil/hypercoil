# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for loss schemes.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from hypercoil.engine.argument import ModelArgument as LossArgument
from hypercoil.loss.scheme import LossScheme
from hypercoil.loss.nn import (
    EntropyLoss,
    EquilibriumLoss,
    CompactnessLoss,
)


class TestLossScheme:

    def test_loss_scheme(self):
        key = jax.random.PRNGKey(0)
        n_groups = 4
        n_channels = 10
        n_dims = 3
        coor = jax.random.normal(key, (n_dims, n_channels))
        weight = jax.random.normal(key, (n_groups, n_channels))
        weight = jax.nn.softmax(weight, axis=-2)

        entropy = EntropyLoss(nu=0.2)
        equilibrium = EquilibriumLoss(nu=20)
        compactness = CompactnessLoss(nu=2, coor=coor)

        loss = LossScheme((entropy, equilibrium, compactness),
                          apply = lambda arg: arg.X)

        def ref(X):
            return entropy(X) + equilibrium(X) + compactness(X)
        ref, g_ref = jax.value_and_grad(ref)(weight)

        arg = LossArgument(X=weight)
        (out, meta), g_out = eqx.filter_jit(eqx.filter_value_and_grad(
            loss, has_aux=True))(arg, key=key)

        assert jnp.allclose(ref, out)
        assert jnp.allclose(g_ref, g_out.X)

        for s in 'Compactness', 'Equilibrium', 'Entropy':
            assert s in meta
