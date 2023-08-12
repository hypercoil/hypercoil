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
from pkg_resources import resource_filename
from hypercoil_examples.synthetic.experiments.run import run_experiment


class TestFrequencyFilterNetwork:
    @pytest.mark.sim
    @pytest.mark.parametrize('expt', [
        'bandident0',
        'bandident1',
        'bandident2',
        'parcellation',
    ])
    def test_fft_band_identification_lowfreq(self, expt):
        results = resource_filename(
            'hypercoil',
            'results'
        )
        run_experiment(
            layer='fft',
            expt=expt,
            results=results,
        )
