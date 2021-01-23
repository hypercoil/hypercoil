# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for atlas map initialisation
"""
import pytest
import torch
import templateflow.api as tflow
from hypernova.init.atlas import (
    ContinuousAtlas,
    DiscreteAtlas,
    atlas_init_
)


class TestAtlasInit:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        path = tflow.get(
            'MNI152NLin6Asym',
            resolution=2,
            atlas='Schaefer2018',
            desc='100Parcels7Networks')
        self.atlas_discrete = DiscreteAtlas(path, mask='auto')
        self.tsr_discrete = torch.empty((
            self.atlas_discrete.n_labels,
            self.atlas_discrete.n_voxels
        ))
        paths = tflow.get(
            'OASIS30ANTs',
            resolution=1,
            suffix='probseg')
        self.atlas_continuous = ContinuousAtlas(paths, mask='auto')
        self.tsr_continuous = torch.empty((
            self.atlas_continuous.n_labels,
            self.atlas_continuous.n_voxels
        ))

    def test_discrete_atlas(self):
        atlas_init_(
            self.tsr_discrete,
            self.atlas_discrete)
        assert torch.allclose(self.tsr_discrete.sum(1), torch.Tensor([1]))
        assert self.tsr_discrete[:, 1].argmax() == 38

    def test_continuous_atlas(self):
        atlas_init_(
            self.tsr_continuous,
            self.atlas_continuous)
        assert torch.allclose(self.tsr_continuous.sum(1), torch.Tensor([1]))
        assert self.tsr_continuous[:, 1].argmax() == 4
