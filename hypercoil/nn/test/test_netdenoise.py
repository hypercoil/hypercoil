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
from hypercoil.examples.synthetic.experiments.run import run_experiment


class TestDenoisingNetwork:
    @pytest.mark.sim
    def test_denoising_homogeneous(self):
        run_experiment(
            layer='denoise',
            expt='homogeneous'
        )

    @pytest.mark.sim
    def test_denoising_heterogeneous(self):
        run_experiment(
            layer='denoise',
            expt='heterogeneous'
        )

    @pytest.mark.sim
    def test_denoising_weakly_correlated(self):
        run_experiment(
            layer='denoise',
            expt='weakcorr'
        )

    @pytest.mark.sim
    def test_denoising_intercorrelated(self):
        run_experiment(
            layer='denoise',
            expt='intercorr'
        )

    @pytest.mark.sim
    def test_denoising_weakly_correlated_elimination(self):
        run_experiment(
            layer='denoise',
            expt='weakcorrelim'
        )

    @pytest.mark.sim
    def test_denoising_intercorrelated_elimination(self):
        run_experiment(
            layer='denoise',
            expt='intercorrelim'
        )

    @pytest.mark.sim
    def test_denoising_batch_50(self):
        run_experiment(
            layer='denoise',
            expt='batch50'
        )

    @pytest.mark.sim
    def test_denoising_batch_25(self):
        run_experiment(
            layer='denoise',
            expt='batch25'
        )

    @pytest.mark.sim
    def test_denoising_batch_12(self):
        run_experiment(
            layer='denoise',
            expt='batch12'
        )
