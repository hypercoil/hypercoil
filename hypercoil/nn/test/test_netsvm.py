# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
SVM learning tests
~~~~~~~~~~~~~~~~~~
Overfitting a simple neural network to test elementary learning capacity for
the SVM module.
"""
import pytest
from hypercoil.synth.experiments.run import run_experiment


class TestSVMNetwork:
    @pytest.mark.sim
    def test_SVM_linear_separation_0(self):
        run_experiment(
            layer='svm',
            expt='linear0'
        )

    @pytest.mark.sim
    def test_SVM_linear_separation_1(self):
        run_experiment(
            layer='svm',
            expt='linear1'
        )

    @pytest.mark.sim
    def test_SVM_radial_collapse(self):
        run_experiment(
            layer='svm',
            expt='rbfcollapse'
        )

    @pytest.mark.sim
    def test_SVM_radial_expansion(self):
        run_experiment(
            layer='svm',
            expt='rbfexpand'
        )
