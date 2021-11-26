# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for brain globe visualisation
"""
import pytest
import matplotlib.pyplot as plt
from pkg_resources import resource_filename as pkgrf
from hypercoil.init.atlas import SurfaceAtlas
from hypercoil.viz.globe import (
    CortexLfsLRFromFiles,
    CortexRfsLRFromFiles,
)


class TestBrainGlobe:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        #TODO
        # Here is some terrible hardcoded pathery
        # When these are added to templateflow, update this
        self.gordon_path = '/mnt/pulsar/Data/atlases/gordon.nii'
        self.glasser_path = '/mnt/pulsar/Data/atlases/glasser.nii'
        self.cmap_modal = pkgrf('hypercoil', 'viz/resources/cmap_modal.nii')
        self.cmap_network = pkgrf('hypercoil', 'viz/resources/cmap_network.nii')

    def test_modal_view(self):
        proj_root = pkgrf('hypercoil', 'results/')
        out_gordon_L = '{}/test_globe_modal_Lgordon.png'.format(proj_root)
        out_glasser_L = '{}/test_globe_modal_Lglasser.png'.format(proj_root)
        out_gordon_R = '{}/test_globe_modal_Rgordon.png'.format(proj_root)
        out_glasser_R = '{}/test_globe_modal_Rglasser.png'.format(proj_root)

        plotter = CortexLfsLRFromFiles(
            data=self.gordon_path,
            cmap=self.cmap_modal,
        )
        plotter()
        plt.savefig(out_gordon_L, bbox_inches='tight')

        plotter = CortexLfsLRFromFiles(
            data=self.glasser_path,
            cmap=self.cmap_modal,
        )
        plotter()
        plt.savefig(out_glasser_L, bbox_inches='tight')

        plotter = CortexRfsLRFromFiles(
            data=self.gordon_path,
            cmap=self.cmap_modal,
        )
        plotter()
        plt.savefig(out_gordon_R, bbox_inches='tight')

        plotter = CortexRfsLRFromFiles(
            data=self.glasser_path,
            cmap=self.cmap_modal,
        )
        plotter()
        plt.savefig(out_glasser_R, bbox_inches='tight')

    def test_network_view(self):
        proj_root = pkgrf('hypercoil', 'results/')
        out_gordon_L = '{}/test_globe_network_Lgordon.png'.format(proj_root)
        out_glasser_L = '{}/test_globe_network_Lglasser.png'.format(proj_root)
        out_gordon_R = '{}/test_globe_network_Rgordon.png'.format(proj_root)
        out_glasser_R = '{}/test_globe_network_Rglasser.png'.format(proj_root)

        plotter = CortexLfsLRFromFiles(
            data=self.gordon_path,
            cmap=self.cmap_network,
        )
        plotter()
        plt.savefig(out_gordon_L, bbox_inches='tight')

        plotter = CortexLfsLRFromFiles(
            data=self.glasser_path,
            cmap=self.cmap_network,
        )
        plotter()
        plt.savefig(out_glasser_L, bbox_inches='tight')

        plotter = CortexRfsLRFromFiles(
            data=self.gordon_path,
            cmap=self.cmap_network,
        )
        plotter()
        plt.savefig(out_gordon_R, bbox_inches='tight')

        plotter = CortexRfsLRFromFiles(
            data=self.glasser_path,
            cmap=self.cmap_network,
        )
        plotter()
        plt.savefig(out_glasser_R, bbox_inches='tight')
