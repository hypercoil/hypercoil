# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for atlas map initialisation
"""
import pytest
import torch
import numpy as np
import nibabel as nb
import templateflow.api as tflow
from nilearn.input_data import NiftiLabelsMasker
from hypernova.nn.atlas import AtlasLinear
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
        self.nil = NiftiLabelsMasker(labels_img=str(path),
                                     resampling_target=None)
        self.aff = nb.load(path).affine
        paths = tflow.get(
            'OASIS30ANTs',
            resolution=1,
            suffix='probseg')
        self.atlas_continuous = ContinuousAtlas(paths, mask='auto')
        self.tsr_continuous = torch.empty((
            self.atlas_continuous.n_labels,
            self.atlas_continuous.n_voxels
        ))
        self.inp = np.arange(91 * 109 * 91 * 50).reshape(91, 109, 91, 50)
        self.inp2 = torch.rand(
            2, 1, 1, 2, *self.atlas_discrete.mask.shape, 10)
        self.inpT = torch.Tensor(self.inp)
        self.lin = AtlasLinear(self.atlas_discrete)

    def test_discrete_atlas(self):
        atlas_init_(
            self.tsr_discrete,
            self.atlas_discrete,
            normalise=True)
        assert torch.allclose(self.tsr_discrete.sum(1), torch.Tensor([1]))
        assert self.tsr_discrete[:, 1].argmax() == 38

    def test_continuous_atlas(self):
        atlas_init_(
            self.tsr_continuous,
            self.atlas_continuous,
            normalise=True)
        assert torch.allclose(self.tsr_continuous.sum(1), torch.Tensor([1]))
        assert self.tsr_continuous[:, 1].argmax() == 4

    def test_atlas_nn_extradims(self):
        out = self.lin(self.inp2)
        assert out.size() == torch.Size([2, 1, 1, 2,
                                         self.atlas_discrete.n_labels, 10])

    def test_atlas_nn_regression(self):
        out = self.lin(self.inpT)
        ref = self.nil.fit_transform(nb.Nifti1Image(self.inp, affine=self.aff))
        assert np.allclose(out.detach().numpy(), ref.T)
