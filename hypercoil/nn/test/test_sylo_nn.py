# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for sylo modules.
"""
import jax
import equinox as eqx
from hypercoil.nn.sylo import Sylo


class TestSylo:
    def test_sylo(self):
        key = jax.random.PRNGKey(0)
        key_d, key_m = jax.random.split(key, 2)
        X = jax.random.normal(key_d, (2, 100, 100))
        model = Sylo(
            in_channels=2,
            out_channels=5,
            dim=100,
            rank=3,
            bias=True,
            symmetry='psd',
            coupling='+',
            fixed_coupling=True,
            remove_diagonal=True,
            key=key_m,
        )
        assert model.templates.shape == (5, 2, 100, 100)
        out = eqx.filter_jit(model)(X)
        assert out.shape == (5, 100, 100)

        X = jax.random.normal(key_d, (2, 100, 50))
        model = Sylo(
            in_channels=2,
            out_channels=5,
            dim=(100, 50),
            rank=3,
            bias=True,
            symmetry=None,
            coupling=0.6,
            fixed_coupling=False,
            remove_diagonal=True,
            key=key_m,
        )
        assert model.templates.shape == (5, 2, 100, 50)
        assert (model.coupling[:3] >= 0).all()
        assert (model.coupling[3:] <= 0).all()
        model = Sylo(
            in_channels=2,
            out_channels=5,
            dim=(100, 50),
            rank=3,
            bias=False,
            symmetry=None,
            coupling='-',
            fixed_coupling=False,
            remove_diagonal=False,
            key=key_m,
        )
        assert (model.coupling <= 0).all()
        out = eqx.filter_jit(model)(X)
        assert out.shape == (5, 100, 50)
