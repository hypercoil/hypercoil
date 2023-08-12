# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Covariance learning tests
~~~~~~~~~~~~~~~~~~~~~~~~~
Overfitting a simple neural network to test elementary learning capacity for
the covariance module.
"""
import pytest
from pkg_resources import resource_filename
from hypercoil_examples.synthetic.experiments.run import run_experiment


class TestCovarianceNetwork:
    @pytest.mark.sim
    @pytest.mark.parametrize('expt', [
        'stateident',
        'parcellationsub',
        'parcellationgroup',
    ])
    def test_cov_state_identification(self, expt):
        results = resource_filename(
            'hypercoil',
            'results'
        )
        run_experiment(
            layer='corr',
            expt=expt,
            results=results,
        )
