# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for sparse matrix utilities.
"""
import jax
import numpy as np
import jax.numpy as jnp
from hypercoil.functional.sparse import(
    random_sparse, spdiagmm, spspmm_full, topk, as_topk, sparse_astype,
    trace_spspmm,
    random_sparse_batchfinal, to_batch_batchfinal, spspmm_batchfinal
)
from hypercoil.functional.utils import vmap_over_outer


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

    def test_spdiagmm(self):
        sp = random_sparse(
            (4, 3, 100, 100),
            k=5,
            key=jax.random.PRNGKey(4839)
        )
        diag = np.random.randn(3, 100)
        diag_embed = jax.vmap(jnp.diagflat, in_axes=(0,))(diag)

        ref_rhsdiag = sp.todense() @ diag_embed
        out_rhsdiag = spdiagmm(sp, diag).todense()
        assert np.allclose(out_rhsdiag, ref_rhsdiag)

        ref_lhsdiag = diag_embed @ sp.todense()
        out_lhsdiag = spdiagmm(diag, sp, lhs_diag=True).todense()
        assert np.allclose(out_lhsdiag, ref_lhsdiag)

    def test_spspmm_full(self):
        sp = random_sparse(
            (4, 3, 100, 100),
            k=5,
            key=jax.random.PRNGKey(4839)
        )
        out = spspmm_full(sp, sp).todense()
        ref = sp.todense() @ sp.todense().swapaxes(-1, -2)
        assert np.allclose(out, ref)

    def test_topk(self):
        def _nlt(x, i): return (x[i] < x).sum()
        X = np.random.randn(2, 3, 1000, 1000)
        k = 5
        out = topk(X, k) #[..., -1]
        out_final = out[..., -1, None]
        nlt = vmap_over_outer(_nlt, 1)((X, out_final))
        assert np.all(nlt == k - 1)

        Xtk = as_topk(X, k, descending=False)
        out = np.where(X <= Xtk.data[..., [-1]], 1, 0).sum(-1)
        assert np.all(out == k)

    def test_sparse_astype(self):
        sp = random_sparse(
            (4, 3, 100, 100),
            k=5,
            key=jax.random.PRNGKey(4839)
        )
        spb = sparse_astype(sp, jnp.bool_)
        assert spb.data.dtype == jnp.bool_
        assert spb.indices.dtype == jnp.int32
        assert spb.dtype == jnp.bool_

    def test_trace_spspmm(self):
        #TODO: there are no correctness tests here.
        sp = random_sparse(
            (1, 3, 100, 100),
            k=5,
            key=jax.random.PRNGKey(4839)
        )

        spb = sparse_astype(sp, jnp.bool_)
        out = trace_spspmm(spb, spb, top_k=False)
        assert out.shape[-1] == 2

        out0 = trace_spspmm(sp, sp, threshold=5, top_k=False,
                            threshold_type='abs>',
                            fix_indices_over_channel_dims=False)
        out1 = trace_spspmm(sp, sp, threshold=5, top_k=False,
                            threshold_type='abs<',
                            fix_indices_over_channel_dims=False)
        assert out0.shape[-1] == out1.shape[-1] == 3
        assert out0.shape[0] + out1.shape[0] == sp.shape[-1] * sp.shape[-2] * sp.shape[-3]

        out0 = trace_spspmm(sp, sp, threshold=0, top_k=False,
                            threshold_type='>',
                            fix_indices_over_channel_dims=False)
        out1 = trace_spspmm(sp, sp, threshold=0, top_k=False,
                            threshold_type='<',
                            fix_indices_over_channel_dims=False)
        out2 = trace_spspmm(spb, spb, top_k=False,
                            fix_indices_over_channel_dims=False)
        assert out0.shape[-1] == out1.shape[-1] == out2.shape[-1] == 3
        assert out0.shape[0] + out1.shape[0] == out2.shape[0]

        sp = random_sparse(
            (4, 3, 1000, 1000),
            k=5,
            key=jax.random.PRNGKey(4839)
        )
        out0 = trace_spspmm(sp, sp, threshold=5, top_k=True)
        assert out0.shape == (1000, 5, 1)
        out0 = trace_spspmm(sp, sp, threshold=5, top_k=True, threshold_type='<')
        assert out0.shape == (1000, 5, 1)
        out0 = trace_spspmm(sp, sp, threshold=5, top_k=True,
                            threshold_type='abs<',
                            fix_indices_over_channel_dims=False)
        assert out0.shape == (3, 1000, 5, 1)
        out0 = trace_spspmm(sp, sp, threshold=5, top_k=True,
                            threshold_type='>',
                            top_k_reduction=None)
        assert out0.shape == (4, 3, 1000, 5, 1)

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
