# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for atlas map initialisation
"""
import pytest
import os
import torch
import numpy as np
import nibabel as nb
import templateflow.api as tflow
import hypercoil
from nilearn.input_data import NiftiLabelsMasker
from hypercoil.nn.atlas import AtlasLinear
from hypercoil.init.atlas import (
    CortexSubcortexCIfTIAtlas,
    DiscreteVolumetricAtlas,
    MultiVolumetricAtlas,
    MultifileVolumetricAtlas,
    DirichletInitVolumetricAtlas,
    DirichletInitSurfaceAtlas,
    _MemeAtlas,
    AtlasInit
)
from hypercoil.functional.noise import UnstructuredDropoutSource


class TestAtlasInit:

    def test_discrete_atlas(self):
        atlas = DiscreteVolumetricAtlas(
            ref_pointer=tflow.get(
                template='MNI152NLin2009cAsym',
                resolution=2,
                desc='100Parcels7Networks',
                suffix='dseg')
        )
        assert atlas.mask.shape[0] == np.prod(atlas.ref.shape)
        assert atlas.mask.sum() == 139951
        assert atlas.compartments['all'].sum() == 139951
        assert len(atlas.decoder['all']) == 100
        assert np.all(atlas.maps['all'].sum(1).numpy() ==
            np.histogram(atlas.cached_ref_data, bins=100, range=(1, 100))[0])
        x, y, z = 84, 62, 13
        assert np.all(
            atlas.coors[97 * 115 * x + 97 * y + z].numpy() / 2 == [x, y, z])

    def test_multivolume_atlas(self):
        atlas = MultiVolumetricAtlas(
            ref_pointer=tflow.get(
                template='MNI152NLin2009cAsym',
                atlas='DiFuMo',
                resolution=2,
                desc='64dimensions')
        )
        assert atlas.mask.shape[0] == np.prod(atlas.ref.shape[:-1])
        assert atlas.mask.sum() == 131238
        assert atlas.compartments['all'].sum() == 131238
        assert len(atlas.decoder['all']) == 64
        assert np.all(
            [atlas.maps['all'][i].max() == atlas.cached_ref_data[..., i].max()
             for i in range(64)])
        x, y, z = 84, 62, 13
        assert np.all(
            atlas.coors[123 * 104 * x + 104 * y + z].numpy() / 2 == [x, y, z])


    def test_multifile_atlas(self):
        atlas = MultifileVolumetricAtlas(
            ref_pointer=[tflow.get(
                template='MNI152NLin2009cAsym',
                suffix='probseg',
                label=l,
                resolution=2)
            for l in ('CSF', 'GM', 'WM')]
        )
        assert atlas.mask.shape[0] == np.prod(atlas.ref.shape[:-1])
        assert atlas.mask.sum() == 281973
        assert atlas.compartments['all'].sum() == 281973
        assert len(atlas.decoder['all']) == 3
        assert np.allclose(atlas.maps['all'].sum(1).numpy(),
            atlas.cached_ref_data.reshape(-1, 3).sum(0))
        x, y, z = 84, 62, 13
        assert np.all(
            atlas.coors[97 * 115 * x + 97 * y + z].numpy() / 2 == [x, y, z])

    def test_cifti_atlas(self):
        atlas = CortexSubcortexCIfTIAtlas(
            ref_pointer='/Users/rastkociric/Downloads/gordon.nii',
            mask_L=tflow.get(
                template='fsLR',
                hemi='L',
                desc='nomedialwall',
                density='32k'),
            mask_R=tflow.get(
                template='fsLR',
                hemi='R',
                desc='nomedialwall',
                density='32k')
        )
        assert atlas.mask.sum() == atlas.ref.shape[-1]
        assert atlas.compartments['cortex_L'].sum() == 29696
        assert atlas.compartments['cortex_R'].sum() == 59412 - 29696
        assert atlas.compartments['cortex_L'].shape == atlas.mask.shape
        assert len(atlas.decoder['cortex_L']) == 161
        assert len(atlas.decoder['cortex_R']) == 172
        assert len(atlas.decoder['subcortex']) == 0
        assert atlas.maps['cortex_L'].shape == (161, 29696)
        assert atlas.maps['cortex_R'].shape == (172, 29716)
        assert atlas.maps['subcortex'].shape == (0,)
        compartment_index = atlas.compartments['cortex_L'][atlas.mask]
        assert np.all(
            atlas.maps['cortex_L'].sum(1).numpy() == np.histogram(
                atlas.cached_ref_data[:, compartment_index],
                bins=360, range=(1, 360)
            )[0][atlas.decoder['cortex_L'] - 1]
        )
        compartment_index = atlas.compartments['cortex_R'][atlas.mask]
        assert np.all(
            atlas.maps['cortex_R'].sum(1).numpy() == np.histogram(
                atlas.cached_ref_data[:, compartment_index],
                bins=360, range=(1, 360)
            )[0][atlas.decoder['cortex_R'] - 1]
        )
        # On a sphere of radius 100
        assert torch.all(
            torch.linalg.norm(atlas.coors[:59412], axis=1).round() == 100)

    def test_compartments_atlas(self):
        atlas = _MemeAtlas()
        assert atlas.mask.shape[0] == np.prod(atlas.ref.shape)
        assert atlas.mask.sum() == 155650
        assert torch.stack(
            (atlas.compartments['eye'], atlas.compartments['face'])
        ).sum(0).bool().sum() == 155650
        assert torch.all(atlas.decoder['eye'] == atlas.decoder['_all'])
        assert atlas.maps['face'].shape == (0,)
        assert np.all(atlas.maps['eye'].sum(-1).numpy() == [1, 5, 1])
        x, y, z = 84, 62, 13
        assert np.all(
            atlas.coors[97 * 115 * x + 97 * y + z].numpy() / 2 == [x, y, z])

    def test_volumetric_dirichlet_atlas(self):
        atlas = DirichletInitVolumetricAtlas(
            mask_source=tflow.get(
                template='MNI152NLin2009cAsym',
                resolution=2,
                desc='brain',
                suffix='mask'
            ),
            compartment_labels={'all': 50}
        )
        assert atlas.mask.sum() == 235840
        assert atlas.decoder['all'].tolist() == list(range(50))
        assert atlas.maps['all'].shape == (50, 235840)
        assert np.allclose(
            torch.softmax(atlas.maps['all'], axis=-2).sum(-2), 1)
        x, y, z = 84, 62, 13
        assert np.all(
            atlas.coors[97 * 115 * x + 97 * y + z].numpy() / 2 == [x, y, z])

    def test_surface_dirichlet_atlas(self):
        atlas = DirichletInitSurfaceAtlas(
            cifti_template='/Users/rastkociric/Downloads/gordon.nii',
            mask_L=tflow.get(
                template='fsLR',
                hemi='L',
                desc='nomedialwall',
                density='32k'),
            mask_R=tflow.get(
                template='fsLR',
                hemi='R',
                desc='nomedialwall',
                density='32k'),
            compartment_labels={
                'cortex_L': 20,
                'cortex_R': 20,
                'subcortex': 20
            }
        )
        assert atlas.mask.sum() == 91282
        assert atlas.decoder['subcortex'].tolist() == list(range(40, 60))
        assert atlas.maps['cortex_L'].shape == (20, 29696)
        assert atlas.maps['cortex_R'].shape == (20, 29716)
        assert atlas.maps['subcortex'].shape == (20, 31870)
        assert atlas.topology['cortex_L'] == 'spherical'
        assert atlas.topology['cortex_R'] == 'spherical'
        assert atlas.topology['subcortex'] == 'euclidean'
        assert np.allclose(
            torch.softmax(atlas.maps['cortex_L'], axis=-2).sum(-2), 1)
        assert np.allclose(
            torch.softmax(atlas.maps['cortex_R'], axis=-2).sum(-2), 1)
        assert np.allclose(
            torch.softmax(atlas.maps['subcortex'], axis=-2).sum(-2), 1)
        # On a sphere of radius 100
        assert torch.all(
            torch.linalg.norm(atlas.coors[:59412], axis=1).round() == 100)

    #TODO: reimplement the below tests. add cuda tests.
    """
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
    """
