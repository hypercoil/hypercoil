# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Sylo learning tests
~~~~~~~~~~~~~~~~~~~
Overfitting a simple neural network to test elementary learning capacity for
the sylo module.
"""
import pytest
from pkg_resources import resource_filename
from hypercoil_examples.synthetic.experiments.run import run_experiment


class TestSyloNetwork:
    @pytest.mark.sim
    @pytest.mark.parametrize('expt', [
        'noreg',
        'reg',
    ])
    def test_sylo_autoencoder_no_regularisation(self, expt):
        results = resource_filename(
            'hypercoil',
            'results'
        )
        run_experiment(
            layer='sylo',
            expt=expt,
            results=results,
        )
