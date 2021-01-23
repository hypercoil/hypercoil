# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for atlas map initialisation
"""
import pytest
import torch
import templateflow.api as tflow
from hypernova.init.atlas import DiscreteAtlas, atlas_init_


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

    def test_discrete_atlas(self):
        atlas_init_(
            self.tsr_discrete,
            self.atlas_discrete)
        assert torch.allclose(self.tsr_discrete.sum(1), torch.Tensor([1]))
        assert self.tsr_discrete[:, 1].argmax() == 38