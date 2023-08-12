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
from hypercoil.examples.synthetic.experiments.run import run_experiment


class TestCovarianceNetwork:
    @pytest.mark.sim
    def test_cov_state_identification(self):
        run_experiment(
            layer='corr',
            expt='stateident'
        )

    @pytest.mark.sim
    def test_cov_state_parcellation_sub(self):
        run_experiment(
            layer='corr',
            expt='parcellationsub'
        )

    @pytest.mark.sim
    def test_cov_state_parcellation_group(self):
        run_experiment(
            layer='corr',
            expt='parcellationgroup'
        )
