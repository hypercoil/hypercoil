# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for QCFC visualisations using netplotbrain
"""
import pytest
import torch
from templateflow import api as tflow
from pkg_resources import resource_filename as pkgrf
from hypercoil.nn import AtlasLinear
from hypercoil.init.atlas import CortexSubcortexCIfTIAtlas
from hypercoil.viz.qcfc import QCFCPlot


class TestQCFCPlot:
    def test_qcfc_plot(self):
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
            dtype=torch.float
        )

        lin = AtlasLinear(atlas)

        qcfc = torch.randn(333, 333)
        qcfc = qcfc @ qcfc.T + 5
        sign = torch.sign(qcfc)
        qcfc = sign * qcfc.abs().sqrt()
        qcfc /= (0.7 * qcfc.max())
        qcfc[qcfc > 1] = 1
        qcfc

        results = pkgrf(
            'hypercoil',
            'results/'
        )
        plotter = QCFCPlot(atlas=lin)
        vec = plotter(qcfc, n=10, significance=0.1, save=f'{results}/qcfc.png')
