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
from hypernova.data.transforms import (
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
