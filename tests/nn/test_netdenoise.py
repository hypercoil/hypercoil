# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Denoising model learning tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Overfitting a simple neural network to test elementary learning capacity for
the denoising model selection module.
"""
import pytest
from pkg_resources import resource_filename
from hypercoil_examples.synthetic.experiments.run import run_experiment


class TestDenoisingNetwork:
    @pytest.mark.sim
    @pytest.mark.parametrize('expt', [
        'homogeneous',
        'heterogeneous',
        'weakcorr',
        'intercorr',
        'weakcorrelim',
        'intercorrelim',
        'batch50',
        'batch25',
        'batch12',
    ])
    def test_denoising_homogeneous(self, expt):
        results = resource_filename(
            'hypercoil',
            'results'
        )
        run_experiment(
            layer='denoise',
            expt=expt,
            results=results,
        )
