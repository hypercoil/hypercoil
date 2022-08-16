# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for sparse matrix utilities.
"""
import jax
import numpy as np
from hypercoil.functional.sparse import(
    random_sparse,
    random_sparse_batchfinal, to_batch_batchfinal, spspmm_batchfinal
)


class TestSparse:
    def test_sparse_topk(self):
        out = random_sparse(
            (4, 3, 1000, 1000),
            k=5,
            key=jax.random.PRNGKey(4839)
        )
        assert out.shape == (4, 3, 1000, 1000)
        assert out.indices.shape == (1, 1, 1000, 5, 1)
        assert out.data.shape == (4, 3, 1000, 5)

    def test_sparse_batch_batchfinal(self):
        shape = (10, 10)
        batch_size = 5
        density = 0.1
        batch = [
            random_sparse_batchfinal(
                jax.random.PRNGKey(np.random.randint(2 ** 32)),
                shape,
                density=density)
            for _ in range(batch_size)]
        B = to_batch_batchfinal(batch)
        nse = int(density * np.prod(shape))
        assert B.shape == shape + (batch_size,)
        assert B.nse <= batch_size * nse
        for M in batch:
            assert M.nse == nse
            assert M.shape == shape

    def test_spspmm_batchfinal(self):
        shape_A = (10, 8)
        shape_B = (10, 18)
        batch_size = 5
        density = 0.1
        batch_A = [
            random_sparse_batchfinal(
                jax.random.PRNGKey(np.random.randint(2 ** 32)),
                shape_A,
                density=density)
            for _ in range(batch_size)]
        batch_B = [
            random_sparse_batchfinal(
                jax.random.PRNGKey(np.random.randint(2 ** 32)),
                shape_B,
                density=density)
            for _ in range(batch_size)]
        A = to_batch_batchfinal(batch_A)
        B = to_batch_batchfinal(batch_B)
        out = spspmm_batchfinal(A, B).todense()
        ref = np.stack([(a.T @ b).todense() for a, b in zip(batch_A, batch_B)], axis=-1)
        assert np.allclose(out, ref)
        assert out.shape == (shape_A[1], shape_B[1], batch_size)

        spspmm_jit = jax.jit(spspmm_batchfinal)
        out = spspmm_jit(A, B)
        assert out.shape == (shape_A[1], shape_B[1], batch_size)
        out = spspmm_jit(A, A)
        assert out.shape == (shape_A[1], shape_A[1], batch_size)
        out = spspmm_jit(B, B)
        assert out.shape == (shape_B[1], shape_B[1], batch_size)
        out = spspmm_jit(B, A)
        assert out.shape == (shape_B[1], shape_A[1], batch_size)
