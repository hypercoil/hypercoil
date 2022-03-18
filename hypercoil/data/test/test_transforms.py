# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Transform tests
~~~~~~~~~~~~~~~
Tests of data transforms atomically and in composition.
"""
import pytest
import torch
import hypercoil.data.functional as F
from hypercoil.data.transforms import (
    Compose, IdentityTransform, EncodeOneHot, ToTensor,
    ApplyModelSpecs, ApplyTransform, BlockTransform,
    UnzipTransformedBlock, ConsolidateBlock,
    ReadDataFrame, ReadNeuroImage
)


class TestTransforms:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.tensor = torch.rand(3, 3, 3)
        self.block1 = [torch.rand(3, 3) for _ in range(2)]
        self.block2 = [[torch.rand(3, 3) for _ in range(2)] for _ in range(3)]
        self.zipped = [[
            {k: torch.rand(3, 3) for k in ('x', 'y', 'z')}
            for _ in range(2)] for _ in range(3)]

    def test_consolidate(self):
        xfm = ConsolidateBlock()
        assert xfm(self.tensor).size() == torch.Size([3, 3, 3])
        assert xfm(self.block1).size() == torch.Size([2, 3, 3])
        assert xfm(self.block2).size() == torch.Size([3, 2, 3, 3])

    def test_unzip(self):
        xfm = Compose([
            UnzipTransformedBlock(),
            ApplyTransform(ConsolidateBlock())
        ])
        assert xfm(self.zipped)['x'].size() == torch.Size([3, 2, 3, 3])

    def test_extend(self):
        tensor_list = [
            torch.rand(1, 3, 3, dtype=torch.double),
            torch.rand(2, 2, 2),
            torch.rand(1, 4, 1)
        ]
        extended_list = F.extend_to_max_size(tensor_list)
        for t in extended_list:
            assert t.shape == (2, 4, 3)

    def test_nanfill(self):
        Z = torch.tensor([
            [0, float('nan')],
            [1, float('nan')],
            [2, 3]
        ])
        Znf, nm = F.nanfill(Z, fill='mean')
        assert torch.logical_not(torch.any(torch.isnan(Znf)))
        assert Znf.mean() == 1.5
        Znf[nm] = -3
        assert Znf.sum() == 0
