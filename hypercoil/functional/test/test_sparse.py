# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for sparse matrix utilities.
"""
import jax
import numpy as np
from hypercoil.functional.sparse import(
    random_sparse, to_batch, spspmm
)


class TestSparse:
    def test_sparse_batch(self):
        shape = (10, 10)
        batch_size = 5
        density = 0.1
        batch = [
            random_sparse(
                jax.random.PRNGKey(np.random.randint(2 ** 32)),
                shape,
                density=density)
            for _ in range(batch_size)]
        B = to_batch(batch)
        nse = int(density * np.prod(shape))
        assert B.shape == shape + (batch_size,)
        assert B.nse <= batch_size * nse
        for M in batch:
            assert M.nse == nse
            assert M.shape == shape

    def test_spspmm(self):
        shape_A = (10, 8)
        shape_B = (10, 18)
        batch_size = 5
        density = 0.1
        batch_A = [
            random_sparse(
                jax.random.PRNGKey(np.random.randint(2 ** 32)),
                shape_A,
                density=density)
            for _ in range(batch_size)]
        batch_B = [
            random_sparse(
                jax.random.PRNGKey(np.random.randint(2 ** 32)),
                shape_B,
                density=density)
            for _ in range(batch_size)]
        A = to_batch(batch_A)
        B = to_batch(batch_B)
        out = spspmm(A, B).todense()
        ref = np.stack([(a.T @ b).todense() for a, b in zip(batch_A, batch_B)], axis=-1)
        assert np.allclose(out, ref)
        assert out.shape == (shape_A[1], shape_B[1], batch_size)

        spspmm_jit = jax.jit(spspmm)
        out = spspmm_jit(A, B)
        assert out.shape == (shape_A[1], shape_B[1], batch_size)
        out = spspmm_jit(A, A)
        assert out.shape == (shape_A[1], shape_A[1], batch_size)
        out = spspmm_jit(B, B)
        assert out.shape == (shape_B[1], shape_B[1], batch_size)
        out = spspmm_jit(B, A)
        assert out.shape == (shape_B[1], shape_A[1], batch_size)
