# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for brain globe visualisation
"""
import pytest
import matplotlib.pyplot as plt
from pkg_resources import resource_filename as pkgrf
from hypercoil.viz.globe import (
    CortexLfsLRFromFiles,
    CortexRfsLRFromFiles,
)


class TestBrainGlobe:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.atlas_path = pkgrf(
            'hypercoil',
            'viz/resources/nullexample.nii'
        )
        self.cmap_modal = pkgrf('hypercoil', 'viz/resources/cmap_modal.nii')
        self.cmap_network = pkgrf('hypercoil', 'viz/resources/cmap_network.nii')

    def test_modal_view(self):
        proj_root = pkgrf('hypercoil', 'results/')
        out_L = '{}/test_globe_modal_L.png'.format(proj_root)
        out_R = '{}/test_globe_modal_R.png'.format(proj_root)

        plotter = CortexLfsLRFromFiles(
            data=self.atlas_path,
            cmap=self.cmap_modal,
        )
        plotter()
        plt.savefig(out_L, bbox_inches='tight')

        plotter = CortexRfsLRFromFiles(
            data=self.atlas_path,
            cmap=self.cmap_modal,
        )
        plotter()
        plt.savefig(out_R, bbox_inches='tight')

    def test_network_view(self):
        proj_root = pkgrf('hypercoil', 'results/')
        out_L = '{}/test_globe_network_L.png'.format(proj_root)
        out_R = '{}/test_globe_network_R.png'.format(proj_root)

        plotter = CortexLfsLRFromFiles(
            data=self.atlas_path,
            cmap=self.cmap_network,
        )
        plotter()
        plt.savefig(out_L, bbox_inches='tight')

        plotter = CortexRfsLRFromFiles(
            data=self.atlas_path,
            cmap=self.cmap_network,
        )
        plotter()
        plt.savefig(out_R, bbox_inches='tight')
