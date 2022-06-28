# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for utility functions.
"""
import torch
import pytest
from hypercoil.functional import (
    apply_mask, wmean, sparse_mm, sparse_rcmul, orient_and_conform
)


class TestUtils:

    def test_wmean(self):
        z = torch.tensor([[
            [1., 4., 2.],
            [0., 9., 1.],
            [4., 6., 7.]],[
            [0., 9., 1.],
            [4., 6., 7.],
            [1., 4., 2.]
        ]])
        w = torch.ones_like(z)
        assert wmean(z, w) == torch.mean(z)
        w = torch.tensor([1., 0., 1.])
        assert torch.all(wmean(z, w, dim=1) == torch.tensor([
            [(1 + 4) / 2, (4 + 6) / 2, (2 + 7) / 2],
            [(0 + 1) / 2, (9 + 4) / 2, (1 + 2) / 2]
        ]))
        assert torch.all(wmean(z, w, dim=2) == torch.tensor([
            [(1 + 2) / 2, (0 + 1) / 2, (4 + 7) / 2],
            [(0 + 1) / 2, (4 + 7) / 2, (1 + 2) / 2]
        ]))
        w = torch.tensor([
            [1., 0., 1.],
            [0., 1., 1.]
        ])
        assert torch.all(wmean(z, w, dim=(0, 1)) == torch.tensor([
            [(1 + 4 + 4 + 1) / 4, (4 + 6 + 6 + 4) / 4, (2 + 7 + 7 + 2) / 4]
        ]))
        assert torch.all(wmean(z, w, dim=(0, 2)) == torch.tensor([
            [(1 + 2 + 9 + 1) / 4, (0 + 1 + 6 + 7) / 4, (4 + 7 + 4 + 2) / 4]
        ]))

    def test_mask(self):
        msk = torch.tensor([1, 1, 0, 0, 0], dtype=torch.bool)
        tsr = torch.rand(5, 5, 5)
        mskd = apply_mask(tsr, msk, axis=0)
        assert mskd.shape == (2, 5, 5)
        assert torch.all(mskd == tsr[:2])
        mskd = apply_mask(tsr, msk, axis=1)
        assert mskd.shape == (5, 2, 5)
        assert torch.all(mskd == tsr[:, :2])
        mskd = apply_mask(tsr, msk, axis=2)
        assert torch.all(mskd == tsr[:, :, :2])
        assert mskd.shape == (5, 5, 2)
        mskd = apply_mask(tsr, msk, axis=-1)
        assert torch.all(mskd == tsr[:, :, :2])
        assert mskd.shape == (5, 5, 2)
        mskd = apply_mask(tsr, msk, axis=-2)
        assert torch.all(mskd == tsr[:, :2])
        assert mskd.shape == (5, 2, 5)
        mskd = apply_mask(tsr, msk, axis=-3)
        assert torch.all(mskd == tsr[:2])
        assert mskd.shape == (2, 5, 5)

    def test_sp_rcmul(self):
        X = torch.rand(20, 3, 4)
        E = torch.randint(2, (2, 20))
        Xs = torch.sparse_coo_tensor(E, X)
        R = torch.sparse.sum(Xs, 1)
        C = torch.sparse.sum(Xs, 0)
        out = sparse_rcmul(Xs, R, C).to_dense()
        ref = (
            R.to_dense().unsqueeze(1) *
            Xs.to_dense() *
            C.to_dense().unsqueeze(0)
        )
        assert torch.allclose(ref, out)

    #TODO: We absolutely need to test this on CUDA.
    def test_sparse_mm(self):
        W = torch.tensor([
            [0.2, -1.3, 2, 0.1, -4],
            [4, 4, 0, -2, -6],
            [0.1, 0., -1, 1, 1]
        ]).t()
        E = torch.tensor([
            [0, 3],
            [0, 4],
            [1, 1],
            [2, 0],
            [3, 2]
        ]).t()
        W.requires_grad = True
        X = torch.sparse_coo_tensor(E, W, size=(5, 5, 3)).coalesce()
        Xd = torch.permute(X.to_dense(), (-1, 0, 1))
        out = torch.permute(
            sparse_mm(X, X.transpose(0, 1)).to_dense(),
            (-1, 0, 1)
        )
        ref = Xd @ Xd.transpose(-1, -2)
        assert torch.allclose(ref, out)
        assert W.grad is None
        out.sum().backward()
        assert W.grad is not None

        W0 = torch.randn(20, 3, 3, 3)
        E0 = torch.stack((
            torch.randint(50, (20,)),
            torch.randint(100, (20,)),
        ))
        X = torch.sparse_coo_tensor(E0, W0, size=(50, 100, 3, 3, 3)).coalesce()
        W1 = torch.randn(20, 3, 3, 3)
        E1 = torch.stack((
            torch.randint(100, (20,)),
            torch.randint(50, (20,)),
        ))
        Y = torch.sparse_coo_tensor(E1, W1, size=(100, 50, 3, 3, 3)).coalesce()
        out = sparse_mm(X, Y)
        assert out.shape == (50, 50, 3, 3, 3)
        ref = (
            X.to_dense().permute(-1, -2, -3, 0, 1) @
            Y.to_dense().permute(-1, -2, -3, 0, 1)
        )
        out = out.to_dense().permute(-1, -2, -3, 0, 1)
        assert torch.allclose(ref, out)

        # And with broadcasting.
        W0 = torch.randn(30, 3, 3, 3)
        E0 = torch.stack((
            torch.randint(50, (30,)),
            torch.randint(100, (30,)),
        ))
        X = torch.sparse_coo_tensor(E0, W0, size=(50, 100, 3, 3, 3)).coalesce()
        W1 = torch.randn(20)
        E1 = torch.stack((
            torch.randint(100, (20,)),
            torch.randint(50, (20,)),
        ))
        Y = torch.sparse_coo_tensor(E1, W1, size=(100, 50)).coalesce()
        out = sparse_mm(X, Y)
        assert out.shape == (50, 50, 3, 3, 3)
        ref = (
            X.to_dense().permute(-1, -2, -3, 0, 1) @
            Y.to_dense()
        )
        out = out.to_dense().permute(-1, -2, -3, 0, 1)
        assert torch.allclose(ref, out)

    def test_orient_and_conform(self):
        X = torch.rand(3, 7)
        R = torch.rand(2, 7, 11, 1, 3)
        out = orient_and_conform(X.transpose(-1, 0), (1, -1), reference=R)
        ref = X.transpose(-1, -2).unsqueeze(-2).unsqueeze(-2).unsqueeze(0)
        assert(out.shape == ref.shape)
        assert torch.all(out == ref)

        X = torch.rand(7)
        R = torch.rand(2, 7, 11, 1, 3)
        out = orient_and_conform(X, 1, reference=R)
        ref = X.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        assert(out.shape == ref.shape)
        assert torch.all(out == ref)
