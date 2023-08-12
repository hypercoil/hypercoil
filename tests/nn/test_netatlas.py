# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pass band learning tests
~~~~~~~~~~~~~~~~~~~~~~~~
Overfitting a simple neural network to test elementary learning capacity for
the atlas module.
"""
import pytest
from pkg_resources import resource_filename
from hypercoil_examples.synthetic.experiments.run import run_experiment


class TestAtlasNetwork:
    @pytest.mark.sim
    @pytest.mark.parametrize('expt', [
        'homology',
        'unsupervisedhard',
        'unsupervisedsoft',
        'hierarchical5',
        'hierarchical25',
    ])
    def test_atlas_homology(self, expt):
        results = resource_filename(
            'hypercoil',
            'results'
        )
        run_experiment(
            layer='atlas',
            expt=expt,
            results=results,
        )
