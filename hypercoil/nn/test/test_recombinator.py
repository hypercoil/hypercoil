# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for recombinator modules.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from hypercoil.nn.recombinator import Recombinator


class TestRecombinator:
    def test_recombinator_fwd(self):
        key = jax.random.PRNGKey(0)
        key_d, key_q, key_m = jax.random.split(key, 3)
        X = jax.random.normal(key_d, (10, 100, 100))
        Q = jax.random.normal(key_q, (10, 10))
        model = Recombinator(10, 2, key=key_m)
        out = eqx.filter_jit(model)(X, query=Q)
        assert out.shape == (2, 100, 100)
        assert not (out >= 0).all()
        model = Recombinator(10, 2, bias=False, positive_only=True, key=key_m)
        out =eqx.filter_jit(model)(jnp.abs(X), query=Q)
        assert out.shape == (2, 100, 100)
        assert (out >= 0).all()
