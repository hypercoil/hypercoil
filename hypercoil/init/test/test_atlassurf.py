# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for atlas surface initialisation

#TODO
These tests are currently deprecated. There was some attempt made to modernise
the nomenclature, but the test represents enough of an edge case that the
current infrastructure just does not support it. Ideally there is no reason we
should not support this case, but our resources are too thinly spread to
support every use case, so for now we've decided to deactivate the test. We
should revisit this after the scaling phase.
"""

"""
import pytest
import torch
import matplotlib.pyplot as plt
from pkg_resources import resource_filename as pkgrf
from hypercoil.init.atlas import (
    BaseAtlas,
    CortexSubcortexCIfTIAtlas,
    _cifti_atlas_common_args
)
from hypercoil.viz.globe import GlobeBrain, GlobeFromFiles
from hypercoil.init.atlasmixins import (
    _CIfTIReferenceMixin,
    _SingleReferenceMixin,
    _FromNullMaskMixin,
    _SingleCompartmentMixin,
    _DiscreteLabelMixin,
    _VertexCIfTIMeshMixin,
    _SpatialConvMixin
)


class TestAtlasSurfInit:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.cifti = pkgrf('hypercoil', 'examples/surf/somatomotor.nii')
        self.gifti = pkgrf('hypercoil', 'examples/surf/somatomotor.gii')

    def test_sphere_conv_init(self):
        class SingleCompartmentCIfTIAtlas(
            _CIfTIReferenceMixin,
            _SingleReferenceMixin,
            _FromNullMaskMixin,
            _SingleCompartmentMixin,
            _DiscreteLabelMixin,
            _VertexCIfTIMeshMixin,
            _SpatialConvMixin,
            BaseAtlas,
        ):
            def __init__(self, ref_pointer, name=None,
                         surf_L=None, surf_R=None,
                         dtype=None, device=None, clear_cache=True,
                         cortex_L='CIFTI_STRUCTURE_CORTEX_LEFT',
                         cortex_R='CIFTI_STRUCTURE_CORTEX_RIGHT'):
                self.surf, _ = _cifti_atlas_common_args(
                    mask_L=None,
                    mask_R=None,
                    surf_L=surf_L,
                    surf_R=surf_R
                )
                print(self.surf)
                super().__init__(ref_pointer=ref_pointer,
                                 mask_source=0,
                                 clear_cache=clear_cache,
                                 name=name,
                                 dtype=dtype,
                                 device=device,
                                 cortex_L=cortex_L,
                                 cortex_R=cortex_R)

        proj_root = pkgrf('hypercoil', 'results/')
        out_ref = '{}/test_atlas_surf_ref.png'.format(proj_root)
        out_map = '{}/test_atlas_surf_map.png'.format(proj_root)
        out_smo = '{}/test_atlas_surf_conv.png'.format(proj_root)
        init = SingleCompartmentCIfTIAtlas(
            ref_pointer=self.cifti,
            name='Somatomotor Neighbourhood',
            surf_L=self.gifti,
            dtype=torch.float
        )

        plotter = GlobeFromFiles(
            data=self.cifti,
            coor=self.gifti,
            shift=(3 * torch.pi / 4),
        )
        plotter()
        plt.savefig(out_ref, bbox_inches='tight')

        map = init(normalise=False)['all']

        print(map)
        print((map[0, :] + map[1, :]).sum())
        print(init.coors.shape)
        print(init.coors)
        raise Exception

        plotter = GlobeBrain(
            data=(map[0, :] + map[1, :]),
            coor=init.coors.T,
            shift=(3 * torch.pi / 4),
            cmap='hot'
        )
        plotter()
        plt.savefig(out_map, bbox_inches='tight')

        map = init(sigma=7, normalise=False)['all']

        plotter = GlobeBrain(
            data=(map[0, :] + map[1, :]),
            coor=init.coors,
            shift=(3 * torch.pi / 4),
            cmap='hot'
        )
        plotter()
        plt.savefig(out_smo, bbox_inches='tight')
"""
