# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for atlas map initialisation
"""
import jax
import numpy as np
import nibabel as nb
import templateflow.api as tflow
from pkg_resources import resource_filename as pkgrf
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker
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
from hypercoil.engine.noise import UnstructuredDropoutSource


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
        assert atlas.mask.size == 139951
        assert atlas.compartments['all'].size == 139951
        assert len(atlas.decoder['all']) == 100
        assert np.all(atlas.maps['all'].sum(1) ==
            np.histogram(atlas.cached_ref_data, bins=100, range=(1, 100))[0])
        x, y, z = 84, 62, 13
        assert np.all(
            atlas.coors[97 * 115 * x + 97 * y + z] / 2 == np.array([x, y, z]))

        img = nb.load(ref_pointer)
        aff = img.affine
        inp = np.linspace(
            0, 1000, np.prod(img.shape) * 20).reshape(
            20, img.shape[1], img.shape[2], img.shape[0]).swapaxes(0, -1)

        nil = NiftiLabelsMasker(labels_img=str(ref_pointer),
                                resampling_target=None)
        ref = nil.fit_transform(nb.Nifti1Image(inp, affine=aff))

        # lin = AtlasLinear(atlas, mask_input=True)
        # out = lin(inp.reshape(-1, 20))
        # assert np.allclose(out.detach().numpy(), ref.T)

    def test_multivolume_atlas(self):
        ref_pointer = tflow.get(
            template='MNI152NLin2009cAsym',
            atlas='DiFuMo',
            resolution=2,
            desc='64dimensions'
        )
        atlas = MultiVolumetricAtlas(
            ref_pointer=ref_pointer,
            clear_cache=False
        )
        assert atlas.mask.shape[0] == np.prod(atlas.ref.shape[:-1])
        assert atlas.mask.size == 131238
        assert atlas.compartments['all'].size == 131238
        assert len(atlas.decoder['all']) == 64
        assert np.all(
            [atlas.maps['all'][i].max() == atlas.cached_ref_data[..., i].max()
             for i in range(64)])
        x, y, z = 84, 62, 13
        assert np.all(
            atlas.coors[123 * 104 * x + 104 * y + z] / 2 == np.array([x, y, z]))

        img = nb.load(ref_pointer)
        aff = img.affine
        inp = np.linspace(
            0, 1000, np.prod(img.shape[:3]) * 5).reshape(
            5, img.shape[1], img.shape[2], img.shape[0]).swapaxes(0, -1)

        nil = NiftiMapsMasker(maps_img=str(ref_pointer),
                              resampling_target=None)
        ref = nil.fit_transform(nb.Nifti1Image(inp, affine=aff))
        # lin = AtlasLinear(
        #     atlas, mask_input=True,
        #     forward_mode='project',
        #     reduction=None)
        # out = lin(inpT.reshape(-1, 5))
        # assert np.allclose(ref.T, out.detach(), rtol=1e-3)

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
        assert atlas.mask.size == 281973
        assert atlas.compartments['all'].size == 281973
        assert len(atlas.decoder['all']) == 3
        assert np.allclose(atlas.maps['all'].sum(1),
            atlas.cached_ref_data.reshape(-1, 3).sum(0))
        x, y, z = 84, 62, 13
        assert np.all(
            atlas.coors[97 * 115 * x + 97 * y + z] / 2 == np.array([x, y, z]))

    def test_cifti_atlas(self):
        ref_pointer = pkgrf(
            'hypercoil',
            'viz/resources/nullexample.nii'
        )
        atlas = CortexSubcortexCIfTIAtlas(
            ref_pointer=ref_pointer,
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
        )
        assert atlas.mask.size == atlas.ref.shape[-1]
        assert atlas.compartments['cortex_L'].size == 29696
        assert atlas.compartments['cortex_R'].size == 59412 - 29696
        assert atlas.compartments['cortex_L'].shape == (29696,)
        assert len(atlas.decoder['cortex_L']) == 200
        assert len(atlas.decoder['cortex_R']) == 200
        assert len(atlas.decoder['subcortex']) == 0
        assert atlas.maps['cortex_L'].shape == (200, 29696)
        assert atlas.maps['cortex_R'].shape == (200, 29716)
        assert atlas.maps['subcortex'].shape == (0,)
        compartment_index = atlas.compartments['cortex_L'].data[atlas.mask.data]
        assert np.all(
            atlas.maps['cortex_L'].sum(1) == np.histogram(
                atlas.cached_ref_data[:, compartment_index],
                bins=400, range=(1, 400)
            )[0][atlas.decoder['cortex_L'] - 1]
        )
        compartment_index = atlas.compartments['cortex_R'].data[atlas.mask.data]
        assert np.all(
            atlas.maps['cortex_R'].sum(1) == np.histogram(
                atlas.cached_ref_data[:, compartment_index],
                bins=400, range=(1, 400)
            )[0][atlas.decoder['cortex_R'] - 1]
        )
        # On a sphere of radius 100
        assert np.all(
            np.linalg.norm(atlas.coors[:59412], axis=1).round() == 100)

        inp = np.random.rand(1, 2, 59412, 3)
        # lin = AtlasLinear(atlas)
        # out = lin.select_compartment('cortex_L', inp)
        # assert out.shape == (1, 2, 29696, 3)

        # out = lin(inp)
        # assert out.shape == (1, 2, 400, 3)

        # lin.decode = True
        # out2 = lin(inp)
        # assert out2.shape == (1, 2, 400, 3)
        # reorder = torch.cat((
        #     lin.atlas.decoder['cortex_L'],
        #     lin.atlas.decoder['cortex_R']
        # ))
        # #assert not torch.allclose(out, out2)
        # assert torch.allclose(out2[..., (reorder - 1), :], out)

        # results = pkgrf(
        #     'hypercoil',
        #     'results/'
        # )
        # atlas.to_image(maps=lin.weight, save=f'{results}/atlas_copy.nii')

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
        assert atlas.mask.size == 155650
        assert np.stack(
            (atlas.compartments['eye'].data, atlas.compartments['face'].data)
        ).sum(0).astype(bool).sum() == 155650
        assert np.all(atlas.decoder['eye'] == atlas.decoder['_all'])
        assert atlas.maps['face'].shape == (0,)
        assert np.all(atlas.maps['eye'].sum(-1) == np.array([1, 5, 1]))
        x, y, z = 84, 62, 13
        assert np.all(
            atlas.coors[97 * 115 * x + 97 * y + z] / 2 == np.array([x, y, z]))

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
            n_labels=50,
            key=jax.random.PRNGKey(0),
        )
        assert atlas.mask.size == 66795
        assert atlas.decoder['all'].tolist() == list(range(1, 51))
        assert atlas.maps['all'].shape == (50, 66795)
        assert np.allclose(
            atlas.maps['all'].sum(-2), 1)
        x, y, z = 84, 62, 13
        assert np.all(
            atlas.coors[97 * 115 * x + 97 * y + z] / 2 == np.array([x, y, z]))

        # lin = AtlasLinear(atlas)
        # out = lin.apply_mask(torch.empty([1, 2, 1082035, 3]))
        # assert out.shape == (1, 2, 66795, 3)

        # out = lin(out)
        # assert out.shape == (1, 2, 50, 3)

        # lin.dropout = UnstructuredDropoutSource(
        #     distr=torch.distributions.Bernoulli(
        #         torch.Tensor([0.2])),
        #     sample_axes=[-1]
        # )
        # empirical = 1 - torch.all(
        #     (lin.postweight['all'] == 0), dim=-2).float().mean()
        # assert (empirical - lin.dropout.distr.mean).abs() < 0.05
        # lin.dropout = None

        # #TODO
        # # Currently we're only testing z-scoring. Add tests for other
        # # reductions.
        # lin.reduction = 'zscore'
        # out = lin(torch.rand(66795, 3))
        # assert np.allclose(out.mean(-1).detach(), 0, atol=1e-3)
        # assert np.allclose(out.std(-1).detach(), 1, atol=1e-3)
        # lin.reduction = 'mean'

    def test_surface_dirichlet_atlas(self):
        cifti_template = pkgrf(
            'hypercoil',
            'viz/resources/nullexample.nii'
        )
        atlas = DirichletInitSurfaceAtlas(
            cifti_template=cifti_template,
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
            },
            key=jax.random.PRNGKey(0),
        )
        assert atlas.mask.size == 59412
        assert atlas.decoder['subcortex'].tolist() == list(range(41, 61))
        assert atlas.maps['cortex_L'].shape == (20, 29696)
        assert atlas.maps['cortex_R'].shape == (20, 29716)
        assert atlas.maps['subcortex'].shape == (20, 0)
        assert atlas.topology['cortex_L'] == 'spherical'
        assert atlas.topology['cortex_R'] == 'spherical'
        assert np.allclose(atlas.maps['cortex_L'].sum(-2), 1)
        assert np.allclose(atlas.maps['cortex_R'].sum(-2), 1)
        assert np.allclose(atlas.maps['subcortex'].sum(-2), 1)
        # On a sphere of radius 100
        assert np.all(
            np.linalg.norm(atlas.coors[:59412], axis=1).round() == 100)

    def test_atlas_empty_compartment(self):
        cifti_template = pkgrf(
            'hypercoil',
            'viz/resources/nullexample.nii'
        )
        atlas = DirichletInitSurfaceAtlas(
            cifti_template=cifti_template,
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
                'cortex_L': 5,
                'cortex_R': 5,
                'subcortex': 0
            },
            key=jax.random.PRNGKey(0),
        )
        # lin = AtlasLinear(atlas)
        # X = torch.rand(lin.mask.sum(), 5)
        # assert lin(X).shape == (10, 5)
