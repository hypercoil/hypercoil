# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for sylo network components
"""
import jax
import jax.numpy as jnp
from hypercoil.functional import sym2vec
from hypercoil.functional.sylo import (
    sylo,
    recombine,
    vertical_compression,
)


#TODO: The tests only check that the code runs, not that it does the right
#      thing.
class TestSylo:
    def test_sylo(self):
        key = jax.random.PRNGKey(0)
        key_x, key_l, key_r, key_c, key_b = jax.random.split(key, 5)
        X = jax.random.normal(key_x, (3, 1, 100, 70))
        L = jax.random.normal(key_l, (5, 1, 100, 2))
        R = jax.random.normal(key_r, (5, 1, 70, 2))
        C = jax.random.normal(key_c, (5, 1, 2, 2))
        C = C + C.swapaxes(-1, -2)
        B = jax.random.normal(key_b, (5,))
        out = sylo(X, L, R, C, bias=B)
        assert out.shape == (3, 5, 100, 70)

        x = jax.random.normal(key_x, (3, 1, 100, 100))
        X = x + x.swapaxes(-1, -2)
        R = jax.random.normal(key_r, (5, 1, 100, 2))
        out = sylo(X, L, R, C, bias=B, symmetry='cross')
        assert out.shape == (3, 5, 100, 100)
        assert jnp.allclose(out[0][0], out[0][0].T, atol=1e-4)
        out = jax.jit(
            sylo, static_argnames=('symmetry', 'remove_diagonal')
        )(X, L, R, C, symmetry='skew', remove_diagonal=True)
        assert out.shape == (3, 5, 100, 100)
        assert jnp.allclose(out[0][0], -out[0][0].T, atol=1e-4)

    def test_recombine(self):
        key = jax.random.PRNGKey(0)
        key_x, key_mix, key_q, key_ql, key_qr = jax.random.split(key, 5)
        X = jax.random.normal(key_x, (3, 10, 30, 40))
        mix = jax.random.normal(key_mix, (5, 10))
        Q = jax.random.normal(key_q, (3, 10, 10))
        QL = jax.random.normal(key_ql, (3, 10, 1))
        QR = jax.random.normal(key_qr, (3, 10, 1))
        out = jax.jit(recombine)(X, mix, Q)
        assert out.shape == (3, 5, 30, 40)
        out = jax.jit(recombine)(X, mix, query_L=QL, query_R=QR)
        assert out.shape == (3, 5, 30, 40)
        Q = QL @ QR.swapaxes(-1, -2)
        out_lr = recombine(X, mix, Q)
        assert jnp.allclose(out, out_lr, atol=1e-4)

    def test_vertical_compression(self):
        key = jax.random.PRNGKey(0)
        key_x, key_r, key_c = jax.random.split(key, 3)
        X = jax.random.normal(key_x, (3, 2, 30, 40))
        R = jax.random.bernoulli(key_r, 0.3, (4, 5, 30))
        C = jax.random.bernoulli(key_c, 0.3, (4, 6, 40))
        out = jax.jit(vertical_compression)(X, R, C)
        assert out.shape == (3, 2, 4, 5, 6)
        X = jax.random.normal(key_x, (3, 2, 30, 30))
        C = jax.random.bernoulli(key_c, 0.3, (4, 5, 30))
        out = jax.jit(
            vertical_compression,
            static_argnames=('renormalise', 'remove_diagonal', 'sign'),
        )(X, R, C, renormalise=True, remove_diagonal=True, sign=-1)
        assert out.shape == (3, 2, 4, 5, 5)
        assert jnp.isclose(sym2vec(out).std(-1).mean(), sym2vec(X).std(-1).mean())
        assert out[1][0][3][2][2] == 0
        out = jax.jit(
            vertical_compression,
            static_argnames=('fold_channels'),
        )(X, R, C, fold_channels=True)
        assert out.shape == (3, 8, 5, 5)
        out = jax.jit(
            vertical_compression,
            static_argnames=('fold_channels'),
        )(X[0], R, C, fold_channels=True)
        assert out.shape == (8, 5, 5)
