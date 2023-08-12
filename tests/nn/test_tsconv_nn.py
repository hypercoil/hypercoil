# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for polynomial convolution layer
"""
import pytest

import jax
import jax.numpy as jnp
import equinox as eqx

from hypercoil.nn import TimeSeriesConv2D, PolyConv2D, BasisConv2D
from hypercoil.init.deltaplus import DeltaPlusInitialiser


class TestPolyConv:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        key = jax.random.PRNGKey(0)
        self.X = jax.random.uniform(key=key, shape=(4, 1, 13, 100))
        self.approx = jnp.allclose

    def test_polyconv_identity(self):
        key = jax.random.PRNGKey(0)
        model = PolyConv2D(degree=2, out_channels=4, key=key)
        model = DeltaPlusInitialiser.init(
            model, loc=(0, 0, 3), var=0, key=key)
        out = model(self.X)
        ref = jnp.tile(self.X, (1, 4, 1, 1))
        assert self.approx(out, ref, atol=1e-5)

    def test_polyconv_shapes(self):
        key = jax.random.PRNGKey(0)
        model = PolyConv2D(degree=7, out_channels=3, key=key)
        out = eqx.filter_jit(model)(self.X).shape
        ref = (4, 3, 13, 100)
        assert out == ref

    def test_basisconv_shapes(self):
        key = jax.random.PRNGKey(0)
        basis_functions = [
            jnp.exp,
            jnp.sin,
            jnp.cos,
            jnp.log,
            jnp.tanh,
        ]
        model = BasisConv2D(
            basis_functions=basis_functions,
            include_const=True,
            out_channels=3,
            key=key
        )
        out = eqx.filter_jit(model)(self.X).shape
        ref = (4, 3, 13, 100)
        assert out == ref

    def test_tsconv_shapes(self):
        key = jax.random.PRNGKey(0)
        model = TimeSeriesConv2D(
            in_channels=1, out_channels=3, key=key)
        out = eqx.filter_jit(model)(self.X).shape
        ref = (4, 3, 13, 100)
        assert out == ref
