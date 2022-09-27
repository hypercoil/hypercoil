# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pass band learning tests
~~~~~~~~~~~~~~~~~~~~~~~~
Overfitting a simple neural network to test elementary learning capacity for
the frequency-domain filtering module.
"""
import pytest
from hypercoil.examples.synthetic.experiments.run import run_experiment


class TestFrequencyFilterNetwork:
    @pytest.mark.sim
    def test_fft_band_identification_lowfreq(self):
        run_experiment(
            layer='fft',
            expt='bandident0'
        )

    @pytest.mark.sim
    def test_fft_band_identification_medfreq(self):
        run_experiment(
            layer='fft',
            expt='bandident1'
        )

    @pytest.mark.sim
    def test_fft_band_identification_highfreq(self):
        run_experiment(
            layer='fft',
            expt='bandident2'
        )

    @pytest.mark.sim
    def test_fft_band_parcellation(self):
        run_experiment(
            layer='fft',
            expt='parcellation'
        )
