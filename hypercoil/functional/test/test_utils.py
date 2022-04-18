# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for utility functions.
"""
import torch
import pytest
from hypercoil.functional import (
    mask, wmean
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
        mskd = mask(tsr, msk, axis=0)
        assert mskd.shape == (2, 5, 5)
        assert torch.all(mskd == tsr[:2])
        mskd = mask(tsr, msk, axis=1)
        assert mskd.shape == (5, 2, 5)
        assert torch.all(mskd == tsr[:, :2])
        mskd = mask(tsr, msk, axis=2)
        assert torch.all(mskd == tsr[:, :, :2])
        assert mskd.shape == (5, 5, 2)
