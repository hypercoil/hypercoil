# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for atlas surface initialisation
"""
import pytest
import torch
import matplotlib.pyplot as plt
from pkg_resources import resource_filename as pkgrf
from hypercoil.init.atlas import SurfaceAtlas
from hypercoil.viz.globe import GlobeBrain, GlobeFromFiles


class TestAtlasSurfInit:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.cifti = pkgrf('hypercoil', 'examples/surf/somatomotor.nii')
        self.gifti = pkgrf('hypercoil', 'examples/surf/somatomotor.gii')

    def test_sphere_conv_init(self):
        proj_root = pkgrf('hypercoil', 'results/')
        out_ref = '{}/test_atlas_surf_ref.png'.format(proj_root)
        out_map = '{}/test_atlas_surf_map.png'.format(proj_root)
        out_smo = '{}/test_atlas_surf_conv.png'.format(proj_root)
        init = SurfaceAtlas(
            path=self.cifti,
            name='Somatomotor Neighbourhood',
            surf_L=self.gifti
        )

        plotter = GlobeFromFiles(
            data=self.cifti,
            coor=self.gifti,
            shift=(3 * torch.pi / 4),
        )
        plotter()
        plt.savefig(out_ref, bbox_inches='tight')

        map = init.map(normalise=False)

        plotter = GlobeBrain(
            data=(map[0, :] + map[1, :]),
            coor=init.dump_coors(),
            shift=(3 * torch.pi / 4),
            cmap='hot'
        )
        plotter()
        plt.savefig(out_map, bbox_inches='tight')

        map = init.map(sigma=7, normalise=False)

        plotter = GlobeBrain(
            data=(map[0, :] + map[1, :]),
            coor=init.dump_coors(),
            shift=(3 * torch.pi / 4),
            cmap='hot'
        )
        plotter()
        plt.savefig(out_smo, bbox_inches='tight')
