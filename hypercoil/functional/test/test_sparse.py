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
    trace_spspmm, _serialised_spspmm, spspmm, _ix, full_as_topk,
    spsp_pairdiff, select_indices, topkx, block_serialise, sp_block_serialise,
    random_sparse_batchfinal, to_batch_batchfinal, spspmm_batchfinal,
    embed_params_in_diagonal, embed_params_in_sparse
)
from hypercoil.functional.utils import vmap_over_outer


class TestSparse:
    def test_block_serialise(self):
        X = np.random.rand(2, 3, 100, 100)
        def f(X, Y): return X.swapaxes(-2, -1) @ Y
        ref = f(X, X)
        f_blk = jax.jit(block_serialise(f, n_blocks=10, out_axes=(-2,)))
        out = f_blk(X, Y=X)
        assert np.allclose(out, ref)

        def g(X, Y): return f(X, Y).sum()
        ref = jax.grad(g)(X, X)
        g_blk_grad = jax.jit(block_serialise(jax.grad(g), n_blocks=10))
        out = g_blk_grad(X, Y=X)
        assert np.allclose(out, ref, atol=1e-5)

        def h(X, Y): return (X * Y).sum()
        Y = np.broadcast_to(np.arange(100), X.shape)
        h_blk_grad = jax.jit(
            block_serialise(
                jax.grad(h), n_blocks=10, argnums=(0, 1)
            ))
        out = h_blk_grad(X, Y)
        assert np.all(out == Y)

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
        out = spspmm_full(sp, sp)
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

    def test_full_as_topk(self):
        U = full_as_topk(np.random.rand(2, 3, 3, 20, 5))
        V = full_as_topk(np.random.rand(2, 3, 3, 20, 5))
        assert U.n_batch == V.n_batch == 4
        assert U.indices.shape == V.indices.shape == (1, 1, 1, 1, 5, 1)
        assert spspmm(U, V).shape == (2, 3, 3, 20, 20)

    def test_topkx(self):
        X = np.random.randn(2, 3, 100, 100)
        f = lambda X: X.swapaxes(-2, -1) @ X
        f_topk = topkx(f, auto_index=True)
        out0 = f_topk(80, X).todense()
        ref = np.where(out0, f(X), 0)
        assert np.allclose(out0, ref)

        indices = select_indices(f(X), 80)
        f_topk = jax.jit(topkx(f))
        out1 = f_topk(indices, X).todense()
        assert np.allclose(out0, out1, atol=1e-5)

    def test_sp_block_serialise(self):
        lhs = random_sparse(
            (4, 3, 1000, 3000),
            k=5,
            key=jax.random.PRNGKey(4839)
        )
        rhs = random_sparse(
            (4, 3, 500, 3000),
            k=5,
            key=jax.random.PRNGKey(4839)
        )
        indices = select_indices(spspmm_full(lhs, rhs), 4)
        f = sp_block_serialise(
            topkx(spspmm_full),
            n_blocks=10,
            argnums=(0,), # indices: non-sparse input to block
            in_axes=(-3,), # blocking axes for indices
            sp_argnums=(1,) # lhs: sparse input to block
        )
        out = f(indices, lhs, rhs=rhs)
        ref = spspmm(lhs, rhs, indices=indices)
        assert np.allclose(out.data, ref.data)
        assert np.all(out.indices == ref.indices)
        assert np.allclose(out.todense(), ref.todense())

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

    def test_spspmm_serialised(self):
        lhs = random_sparse(
            (4, 3, 1000, 3000),
            k=5,
            key=jax.random.PRNGKey(4839)
        )
        rhs = random_sparse(
            (4, 3, 500, 3000),
            k=5,
            key=jax.random.PRNGKey(4839)
        )
        indices = trace_spspmm(lhs, rhs, threshold=5, top_k=True)
        out = _serialised_spspmm(lhs, rhs, indices, n_blocks=10)
        assert indices.max() <= 500
        assert out.shape == (4, 3, 1000, 500)
        assert out.data.shape == (4, 3, 1000, 5)
        assert out.indices.shape == (1, 1, 1000, 5, 1)
        assert out.n_batch == 3

        sspspmm_jit = jax.jit(
            _serialised_spspmm,
            static_argnames=('n_blocks',)
        )
        out = sspspmm_jit(lhs, rhs, indices, 10)
        assert out.shape == (4, 3, 1000, 500)
        assert out.data.shape == (4, 3, 1000, 5)
        assert out.indices.shape == (1, 1, 1000, 5, 1)
        assert out.n_batch == 3

    def test_spspmm_topk(self):
        key = jax.random.PRNGKey(4839)
        k0, k1 = jax.random.split(key)
        lhs = random_sparse(
            (4, 3, 1000, 3000),
            k=5,
            key=k0
        )
        rhs = random_sparse(
            (4, 3, 500, 3000),
            k=5,
            key=k1
        )
        indices = trace_spspmm(lhs, rhs, threshold=5, top_k=True)

        spspmm_jit = jax.jit(spspmm, static_argnames=('n_blocks',))
        out0 = spspmm_jit(lhs, rhs)
        assert out0.shape == (4, 3, 1000, 500)
        sampling_fn = vmap_over_outer(_ix, 1)
        out0_data = sampling_fn((
            out0,
            indices.squeeze(-1)
        ))

        out1 = spspmm_jit(lhs, rhs, indices)
        assert out1.shape == (4, 3, 1000, 500)
        assert np.allclose(out0_data, out1.data)

        out2 = spspmm_jit(lhs, rhs, indices, n_blocks=10)
        assert out2.shape == (4, 3, 1000, 500)
        assert np.allclose(out0_data, out2.data)

    def test_spsppairdiff(self):
        key = jax.random.PRNGKey(4839)
        k0, k1 = jax.random.split(key)
        lhs = random_sparse(
            (4, 3, 100, 300),
            k=5,
            key=k0
        )
        rhs = random_sparse(
            (4, 3, 50, 300),
            k=5,
            key=k1
        )
        out = spsp_pairdiff(lhs, rhs)
        ref = lhs.todense()[..., None, :] - rhs.todense()[..., None, :, :]
        assert np.all(out.todense() == ref)

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

    def test_params_embed(self):
        params = np.random.rand(2, 3, 2, 10)
        out = embed_params_in_diagonal(params)
        assert out.shape == (2, 3, 2, 10, 10)
        ref = vmap_over_outer(jnp.diagflat, 1)((params,))
        assert np.allclose(out.todense(), ref)

        Q = np.random.randn(2, 3, 2, 10, 10)
        Q = np.where(Q > 0.8, Q, 0)
        assert np.all(embed_params_in_sparse(Q).todense() == Q)

        Q = np.random.randn(10, 10)
        Q = np.where(Q > 0.8, Q, 0)
        assert np.all(embed_params_in_sparse(Q).todense() == Q)
