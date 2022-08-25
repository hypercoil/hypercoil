# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for base initialisers
"""
import jax
import equinox as eqx
import numpy as np
from distrax import Normal
from hypercoil.init.base import (
    DistributionInitialiser, ConstantInitialiser, IdentityInitialiser
)
from hypercoil.init.mapparam import MappedLogits, _to_jax_array


class TestBaseInit:

    def test_distribution_init(self):
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(key=key, in_features=2, out_features=3)
        model = DistributionInitialiser.init(
            model, distribution=Normal(loc=100, scale=0.1), key=key)
        assert np.abs(model.weight - 100).mean() < 0.1

        model = DistributionInitialiser.init(
            model,
            distribution=Normal(loc=0.5, scale=0.01),
            mapper=MappedLogits,
            key=key
        )
        assert np.abs(_to_jax_array(model.weight) - 0.5).mean() < 0.05
        assert np.abs(model.weight.original - 0).mean() < 0.1

    def test_constant_init(self):
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(key=key, in_features=2, out_features=3)
        model = ConstantInitialiser.init(model, value=1.)
        assert np.all(model.weight == 1)

    def test_identity_init(self):
        key = jax.random.PRNGKey(0)
        model = eqx.nn.Linear(key=key, in_features=5, out_features=5)
        model = IdentityInitialiser.init(model, scale=-1., shift=1.)
        assert np.all(model.weight == ~np.eye(5, dtype=bool))
