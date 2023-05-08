# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for atlas map initialisation
"""
from pkg_resources import resource_filename as pkgrf
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import nibabel as nb
import templateflow.api as tflow
from hypercoil.nn.atlas import AtlasLinear
from hypercoil.init.atlas import (
    BaseAtlas,
    CortexSubcortexCIfTIAtlas,
    CortexSubcortexGIfTIAtlas,
)
from hypercoil.init.atlasmixins import (
    Reference,
    _GIfTIReferenceMixin,
    _CortexSubcortexCIfTIMaskMixin
)


class TestAtlasInit:

    def test_gifti_atlas(self):
        ref_L = pkgrf(
            'hypercoil',
            'viz/resources/nullexample_L.gii'
        )
        ref_R = pkgrf(
            'hypercoil',
            'viz/resources/nullexample_R.gii'
        )
        ref_pointer = (ref_L, ref_R, None,)

        mask_source = {
            'mask_L': tflow.get(
                template='fsLR',
                hemi='L',
                desc='nomedialwall',
                density='32k'
            ),
            'mask_R': tflow.get(
                template='fsLR',
                hemi='R',
                desc='nomedialwall',
                density='32k'
            ),
        }

        r = _GIfTIReferenceMixin()
        m = _CortexSubcortexCIfTIMaskMixin()

        ref = r._load_reference(ref_pointer)
        mask = m._create_mask(mask_source)

        atlas = CortexSubcortexGIfTIAtlas(
            data_L=ref_pointer[0],
            data_R=ref_pointer[1],
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
        compartment_index = atlas.compartments['cortex_L'].data

        results = pkgrf(
            'hypercoil',
            'results/'
        )
        atlas.to_gifti(save=f'{results}/atlas_copy_gifti')


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
        compartment_index = atlas.compartments['cortex_L'].data
        assert np.all(
            atlas.maps['cortex_L'].sum(1) == np.histogram(
                atlas.ref.dataobj[:, compartment_index],
                bins=400, range=(1, 400)
            )[0][atlas.decoder['cortex_L'] - 1]
        )
        compartment_index = atlas.compartments['cortex_R'].data
        assert np.all(
            atlas.maps['cortex_R'].sum(1) == np.histogram(
                atlas.ref.dataobj[:, compartment_index],
                bins=400, range=(1, 400)
            )[0][atlas.decoder['cortex_R'] - 1]
        )
        # On a sphere of radius 100
        assert np.all(
            np.linalg.norm(atlas.coors[:59412], axis=1).round() == 100)

        inp = np.random.rand(1, 2, 59412, 3)
        lin = AtlasLinear.from_atlas(atlas=atlas, key=jax.random.PRNGKey(0))
        out = lin(inp)
        assert out.shape == (1, 2, 400, 3)
        reorder = jnp.concatenate((
            lin.decoder['cortex_L'],
            lin.decoder['cortex_R'],
        ))
        assert np.allclose(out[..., (reorder - 1), :], out)

        results = pkgrf(
            'hypercoil',
            'results/'
        )
        atlas.to_cifti(maps=lin.weight, save=f'{results}/atlas_copy2.nii')
        atlas.to_gifti(save=f'{results}/atlas_copy_gifti_from_cifti')

