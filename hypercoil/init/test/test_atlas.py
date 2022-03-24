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
    AtlasInit,
)
from hypercoil.init.atlasmixins import (
    MaskThreshold,
    MaskNegation,
    MaskIntersection
)
from hypercoil.functional.noise import UnstructuredDropoutSource


class TestAtlasInit:

    def test_discrete_atlas(self):
        ref_pointer = tflow.get(
            template='MNI152NLin2009cAsym',
            resolution=2,
            desc='100Parcels7Networks',
            suffix='dseg'
        )
        atlas = DiscreteVolumetricAtlas(
            ref_pointer=ref_pointer,
            clear_cache=False
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

        img = nb.load(ref_pointer)
        aff = img.affine
        inp = inp = np.linspace(
            0, 1000, np.prod(img.shape) * 20).reshape(
            20, img.shape[1], img.shape[2], img.shape[0]).swapaxes(0, -1)
        inpT = torch.tensor(inp, dtype=torch.float)

        nil = NiftiLabelsMasker(labels_img=str(ref_pointer),
                                resampling_target=None)
        ref = nil.fit_transform(nb.Nifti1Image(inp, affine=aff))

        lin = AtlasLinear(atlas, mask_input=True)
        out = lin(inpT.reshape(-1, 20))
        assert np.allclose(out.detach().numpy(), ref.T)

    def test_multivolume_atlas(self):
        atlas = MultiVolumetricAtlas(
            ref_pointer=tflow.get(
                template='MNI152NLin2009cAsym',
                atlas='DiFuMo',
                resolution=2,
                desc='64dimensions'),
            clear_cache=False
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
            for l in ('CSF', 'GM', 'WM')],
            clear_cache=False
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
                density='32k'),
            clear_cache=False,
            dtype=torch.float
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

        inp = torch.rand([1, 2, 91282, 3])
        lin = AtlasLinear(atlas)
        out = lin.select_compartment('cortex_L', inp)
        assert out.shape == (1, 2, 29696, 3)

        out = lin(inp)
        assert out.shape == (1, 2, 333, 3)

        lin.decode = True
        out2 = lin(inp)
        assert out2.shape == (1, 2, 333, 3)
        reorder = torch.cat((
            lin.atlas.decoder['cortex_L'],
            lin.atlas.decoder['cortex_R']
        ))
        assert not torch.allclose(out, out2)
        assert torch.allclose(out2[..., (reorder - 1), :], out)

        """
        Let's keep this on CUDA only. It's extremely slow.
        from hypercoil.functional.cov import pairedcorr
        maps = atlas(
            compartments=['cortex_L'],
            normalise=True,
            sigma=3,
            truncate=20,
            max_bin=1000
        )
        assert np.allclose(maps['cortex_L'].sum(-1), 1)
        assert pairedcorr(
            maps['cortex_L'][0].view(1, -1),
            atlas.maps['cortex_L'][0].view(1, -1)
        ) > 0.9
        """

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
            mask_source=MaskIntersection(
                MaskThreshold(
                    tflow.get(
                        template='MNI152NLin2009cAsym',
                        resolution=2,
                        label='GM',
                        suffix='probseg'
                    ),
                    threshold=0.5
                ),
                MaskNegation(MaskThreshold(
                    tflow.get(
                        template='MNI152NLin2009cAsym',
                        resolution=2,
                        label='WM',
                        suffix='probseg'
                    ),
                    threshold=0.2
                )),
                MaskNegation(MaskThreshold(
                    tflow.get(
                        template='MNI152NLin2009cAsym',
                        resolution=2,
                        label='CSF',
                        suffix='probseg'
                    ),
                    threshold=0.2
                ))
            ),
            n_labels=50
        )
        assert atlas.mask.sum() == 66795
        assert atlas.decoder['all'].tolist() == list(range(50))
        assert atlas.maps['all'].shape == (50, 66795)
        assert np.allclose(
            torch.softmax(atlas.maps['all'], axis=-2).sum(-2), 1)
        x, y, z = 84, 62, 13
        assert np.all(
            atlas.coors[97 * 115 * x + 97 * y + z].numpy() / 2 == [x, y, z])

        lin = AtlasLinear(atlas)
        out = lin.apply_mask(torch.empty([1, 2, 1082035, 3]))
        assert out.shape == (1, 2, 66795, 3)

        out = lin(out)
        assert out.shape == (1, 2, 50, 3)

        lin.dropout = UnstructuredDropoutSource(
            distr=torch.distributions.Bernoulli(
                torch.Tensor([0.2])),
            sample_axes=[-1]
        )
        empirical = 1 - torch.all(
            (lin.postweight['all'] == 0), dim=-2).float().mean()
        assert (empirical - lin.dropout.distr.mean).abs() < 0.05
        lin.dropout = None

        #TODO
        # Currently we're only testing z-scoring. Add tests for other
        # reductions.
        lin.reduction = 'zscore'
        out = lin(torch.rand(66795, 3))
        assert np.allclose(out.mean(-1).detach(), 0, atol=1e-4)
        assert np.allclose(out.std(-1).detach(), 1, atol=1e-4)
        lin.reduction = 'mean'

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

    @pytest.mark.cuda
    def test_cifti_atlas_cuda(self):
        atlas = CortexSubcortexCIfTIAtlas(
            ref_pointer='/home/rastko/Downloads/atlases/gordon.nii',
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
            clear_cache=False,
            dtype=torch.float,
            device='cuda'
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
        compartment_index = atlas.compartments['cortex_L'][atlas.mask].cpu()
        assert np.all(
            atlas.maps['cortex_L'].sum(1).cpu().numpy() == np.histogram(
                atlas.cached_ref_data[:, compartment_index],
                bins=360, range=(1, 360)
            )[0][atlas.decoder['cortex_L'] - 1]
        )
        compartment_index = atlas.compartments['cortex_R'][atlas.mask].cpu()
        assert np.all(
            atlas.maps['cortex_R'].sum(1).cpu().numpy() == np.histogram(
                atlas.cached_ref_data[:, compartment_index],
                bins=360, range=(1, 360)
            )[0][atlas.decoder['cortex_R'] - 1]
        )
        # On a sphere of radius 100
        assert torch.all(
            torch.linalg.norm(atlas.coors[:59412], axis=1).round() == 100)

        inp = torch.rand([1, 2, 91282, 3], device='cuda')
        lin = AtlasLinear(atlas, device='cuda')
        out = lin.select_compartment('cortex_L', inp)
        assert out.shape == (1, 2, 29696, 3)

        out = lin(inp)
        assert out.shape == (1, 2, 333, 3)

        lin.decode = True
        out2 = lin(inp)
        assert out2.shape == (1, 2, 333, 3)
        reorder = torch.cat((
            lin.atlas.decoder['cortex_L'],
            lin.atlas.decoder['cortex_R']
        ))
        assert not torch.allclose(out, out2)
        assert torch.allclose(out2[..., (reorder - 1), :], out)

        #Let's keep this on CUDA only. It's extremely slow.
        from hypercoil.functional.cov import pairedcorr
        maps = atlas(
            compartments=['cortex_L'],
            normalise=True,
            sigma=3,
            truncate=20,
            max_bin=1000
        )
        assert maps['cortex_L'].device.type == 'cuda'
        assert np.allclose(maps['cortex_L'].sum(-1).cpu(), 1)
        assert pairedcorr(
            maps['cortex_L'][0].view(1, -1),
            atlas.maps['cortex_L'][0].view(1, -1)
        ) > 0.9
