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
from hypercoil.nn.atlas import AtlasLinear
from hypercoil.init.atlas import (
    MultifileAtlas,
    MultivolumeAtlas,
    DiscreteAtlas,
    AtlasInit
)
from hypercoil.functional.noise import UnstructuredDropoutSource


class TestAtlasInit:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        path = tflow.get(
            'MNI152NLin6Asym',
            resolution=2,
            atlas='Schaefer2018',
            desc='100Parcels7Networks')
        self.atlas_discrete = DiscreteAtlas(path, mask='auto')
        self.tsr_discrete = torch.nn.ParameterDict({
            'all' : torch.nn.Parameter(torch.empty((
                self.atlas_discrete.n_labels['_all_compartments'],
                self.atlas_discrete.n_voxels
            )))
        })
        self.nil = NiftiLabelsMasker(labels_img=str(path),
                                     resampling_target=None)
        self.aff = nb.load(path).affine
        paths = tflow.get(
            'OASIS30ANTs',
            resolution=1,
            suffix='probseg')
        self.atlas_continuous = MultifileAtlas(paths, mask='auto')
        self.tsr_continuous = torch.nn.ParameterDict({
            'all' : torch.nn.Parameter(torch.empty((
                self.atlas_continuous.n_labels['_all_compartments'],
                self.atlas_continuous.n_voxels
            )))
        })
        self.inp = np.linspace(
            0, 1000, 91 * 109 * 91 * 50).reshape(
            50, 109, 91, 91).swapaxes(0, -1)
        self.inp2 = torch.rand(
            2, 1, 1, 2, *self.atlas_discrete.mask.shape, 10)
        self.inpT = torch.Tensor(self.inp)
        #self.lin = AtlasLinear(self.atlas_discrete)

    def test_discrete_atlas(self):
        init = AtlasInit(
            self.atlas_discrete,
            normalise=True
        )
        init(self.tsr_discrete)
        map = self.tsr_discrete['all']
        assert torch.allclose(map.sum(1), torch.Tensor([1]))
        assert map[:, 1].argmax() == 38

    def test_continuous_atlas(self):
        init = AtlasInit(
            self.atlas_continuous,
            normalise=True
        )
        init(self.tsr_continuous)
        map = self.tsr_continuous['all']
        assert torch.allclose(map.sum(1), torch.Tensor([1]))
        assert map[:, 1].argmax() == 4

    def test_atlas_nn_extradims(self):
        out = self.lin(self.inp2)
        assert out.size() == torch.Size([2, 1, 1, 2,
                                         self.atlas_discrete.n_labels, 10])

    def test_atlas_nn_regression(self):
        out = self.lin(self.inpT)
        ref = self.nil.fit_transform(nb.Nifti1Image(self.inp, affine=self.aff))
        assert np.allclose(out.detach().numpy(), ref.T)

    def test_atlas_nn_reductions(self):
        #TODO
        # Currently we're only testing z-scoring. Add tests for other
        # reductions.
        self.lin.reduction = 'zscore'
        out = self.lin(self.inpT)
        assert np.allclose(out.mean(-1).detach(), 0, atol=1e-5)
        assert np.allclose(out.std(-1).detach(), 1, atol=1e-5)
        self.lin.reduction = 'mean'

    def test_atlas_nn_dropout(self):
        self.lin.dropout = UnstructuredDropoutSource(
            distr=torch.distributions.Bernoulli(
                torch.Tensor([0.2])),
            sample_axes=[-1]
        )
        empirical = 1 - torch.all(
            self.lin.postweight==0, dim=-2).float().mean()
        assert (empirical - self.lin.dropout.distr.mean).abs() < 0.05
        self.lin.dropout = None
