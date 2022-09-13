# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for vertical compression modules.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from hypercoil.nn.vertcom import (
    VerticalCompression,
    random_bipartite_lattice,
)


class TestVerticalCompression:
    def test_random_lattice(self):
        key = jax.random.PRNGKey(0)
        conditions = {
            'in_features': (10, 20),
            'out_features': (4, 5),
            'lattice_order': (1, 2, 3, 4),
        }
        for i in conditions['in_features']:
            for o in conditions['out_features']:
                for l in conditions['lattice_order']:
                    key = jax.random.split(key, 1)[0]
                    lattice = random_bipartite_lattice(l, i, o, key=key)
                    assert lattice.shape == (o, i)
                    assert (lattice >= 0).all()
                    assert (lattice <= 1).all()
                    lcm = jnp.lcm(i, o)
                    n_edges = jnp.sum(lattice)
                    assert n_edges == min(l * lcm, i * o)
                    assert (lattice.sum(0) == n_edges / i).all()
                    assert (lattice.sum(1) == n_edges / o).all()

    def test_vertical_compression(self):
        key = jax.random.PRNGKey(0)
        key_d, key_m = jax.random.split(key, 2)
        X = jax.random.normal(key_d, (2, 100, 100))
        model = VerticalCompression(100, 10, 2, 2, key=key_m)
        assert (model.mask.sum(0) == 2).all()
        assert (model.mask.sum(1) == 20).all()
        out = eqx.filter_jit(model)(X)
        assert out.shape == (4, 10, 10)
        model = VerticalCompression.mode(model, 'uncompress')
        out = eqx.filter_jit(model)(out)
        assert out.shape == (8, 100, 100)
        model = VerticalCompression.mode(model, 'reconstruct')
        out = eqx.filter_jit(model)(X)
        assert out.shape == (8, 100, 100)
