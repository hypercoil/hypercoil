# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for parameter address grammar
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any
from hypercoil.formula.nnops import (
    ParameterAddressGrammar,
)


class ModuleWithSubmodules(eqx.Module):
    sub0: Any
    sub1: Any


class TestAddress:

    @staticmethod
    def model_zero(f, model):
        return eqx.tree_at(
            f,
            model,
            replace_fn=jnp.zeros_like
        )

    @staticmethod
    def param_zero_assert(f, model_test, model_ref):
        assert (f(model_test) == jnp.zeros_like(f(model_ref))).all()

    def test_trivial(self):
        grammar = ParameterAddressGrammar()

        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(in_features=1, out_features=1, key=key)
        search_str = 'weight'
        f = grammar.compile(search_str)
        model_zero = self.model_zero(f, model)
        self.param_zero_assert(
            lambda m: m.weight, model_zero, model)

    def test_address(self):
        grammar = ParameterAddressGrammar()

        key = jax.random.PRNGKey(0)
        k0, k1, k2, k3, k4, k5, k6, k7, k8, k9, = jax.random.split(key, 10)
        model_0 = eqx.nn.Linear(in_features=4, out_features=2, key=k0)
        model_1 = eqx.nn.MLP(3, 2, 1, depth=7, key=k1)
        model_2 = ModuleWithSubmodules(
            sub0 = eqx.nn.Linear(in_features=2, out_features=3, key=k2),
            sub1 = eqx.nn.Linear(in_features=3, out_features=3, key=k3),
        )
        model = eqx.nn.Sequential((
            eqx.nn.Linear(in_features=1, out_features=3, key=k4),
            {
                'a': eqx.nn.Linear(in_features=5, out_features=2, key=k5),
                'b': model_0,
            },
            model_1,
            eqx.nn.Linear(in_features=2, out_features=2, key=k6),
            model_2,
            eqx.nn.Linear(in_features=4, out_features=1, key=k7),
            eqx.nn.Linear(in_features=5, out_features=1, key=k8),
            eqx.nn.Linear(in_features=6, out_features=1, key=k9),
        ))
        
        search_str = '#0.bias'
        f = grammar.compile(search_str)
        model_zero = self.model_zero(f, model)
        self.param_zero_assert(
            f=lambda m: m[0].bias,
            model_test=model_zero,
            model_ref=model
        )

        search_str = '#4.sub1.weight'
        f = grammar.compile(search_str)
        model_zero = self.model_zero(f, model)
        self.param_zero_assert(
            f=lambda m: m[4].sub1.weight,
            model_test=model_zero,
            model_ref=model
        )

        search_str = '#1$b.weight'
        f = grammar.compile(search_str)
        f(model)
        model_zero = self.model_zero(f, model)
        self.param_zero_assert(
            f=lambda m: m[1]['b'].weight,
            model_test=model_zero,
            model_ref=model
        )

        search_str = '#1$(a;b).weight'
        f = grammar.compile(search_str)
        f(model)
        model_zero = self.model_zero(f, model)
        self.param_zero_assert(
            f=lambda m: m[1]['a'].weight,
            model_test=model_zero,
            model_ref=model)
        self.param_zero_assert(
            f=lambda m: m[1]['b'].weight,
            model_test=model_zero,
            model_ref=model)

        search_str = '#5:.weight'
        f = grammar.compile(search_str)
        f(model)
        model_zero = self.model_zero(f, model)
        self.param_zero_assert(
            f=lambda m: m[5].weight,
            model_test=model_zero,
            model_ref=model)
        self.param_zero_assert(
            f=lambda m: m[6].weight,
            model_test=model_zero,
            model_ref=model)
        self.param_zero_assert(
            f=lambda m: m[7].weight,
            model_test=model_zero,
            model_ref=model)
