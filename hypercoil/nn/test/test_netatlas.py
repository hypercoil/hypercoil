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
from hypercoil.synth.experiments.run import run_experiment


class TestAtlasNetwork:
    @pytest.mark.sim
    def test_atlas_homology(self):
        run_experiment(
            layer='atlas',
            expt='homology'
        )

    @pytest.mark.sim
    def test_atlas_unsupervised_hard(self):
        run_experiment(
            layer='atlas',
            expt='unsupervisedhard'
        )

    @pytest.mark.sim
    def test_atlas_unsupervised_soft(self):
        run_experiment(
            layer='atlas',
            expt='unsupervisedsoft'
        )

    @pytest.mark.sim
    def test_atlas_hierarchical_5(self):
        run_experiment(
            layer='atlas',
            expt='hierarchical5'
        )

    @pytest.mark.sim
    def test_atlas_hierarchical_25(self):
        run_experiment(
            layer='atlas',
            expt='hierarchical25'
        )
