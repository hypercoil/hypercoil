# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Pass band learning tests
~~~~~~~~~~~~~~~~~~~~~~~~
Overfitting a simple neural network to test elementary learning capacity for
the frequency-domain filtering module.
"""
import os
import pytest
import torch
import hypercoil
import numpy as np
import matplotlib.pyplot as plt
from hypercoil.nn import (
    FrequencyDomainFilter,
    UnaryCovarianceUW
)
from hypercoil.functional import corr
from hypercoil.init import FreqFilterSpec
from hypercoil.functional.activation import complex_decompose
from hypercoil.functional.domain import Identity
from hypercoil.functional.noise import UnstructuredDropoutSource
from hypercoil.reg import (
    SmoothnessPenalty, SymmetricBimodal,
    RegularisationScheme, NormedRegularisation
)
from .synth_netbp import synthesise
from .overfit_plot import overfit_and_plot_progress


TEST_BAND = 2


class TestFrequencyFilterNetwork:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.n = 1000
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        np.random.seed(0)
        self.X, y, target = synthesise(n=self.n)
        self.y = y[TEST_BAND]
        self.target = FreqFilterSpec(Wn=target[TEST_BAND], ftype='ideal')
        self.max_epoch = 300
        self.loss = torch.nn.MSELoss()
        self.log_interval = 5
        self.max_score = np.sqrt(.01 * self.n)
        self.dim = self.n // 2 + 1

    def test_supervised_bp_network(self):
        out = '{}/results/test_supervised_bp_network.svg'.format(
            os.path.dirname(hypercoil.__file__))

        torch.manual_seed(0)
        filter_specs = [FreqFilterSpec(
            Wn=None, ftype='randn', btype=None, # clamps = [{0: 0}]
            ampl_scale=0.01, phase_scale=0.01
        )]

        survival_prob=0.2
        drop = UnstructuredDropoutSource(
            distr=torch.distributions.Bernoulli(survival_prob),
            training=True
        )

        model = torch.nn.Sequential(
            FrequencyDomainFilter(
                dim=self.dim,
                filter_specs=filter_specs,
                domain=Identity()
            ),
            UnaryCovarianceUW(
                dim=self.n,
                estimator=corr,
                dropout=drop
            )
        )

        # SGD tends to get stuck in worse minima here
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        reg_ampl = RegularisationScheme([
            SmoothnessPenalty(nu=0.2),
            SymmetricBimodal(nu=0.05),
            NormedRegularisation(nu=0.015)
        ])
        # Phase reg doesn't seem to work . . .
        # We get phase randomisation where the amplitude is close to zero
        # and placing a too strict penalty on phase for some reason messes
        # up the amplitude structure

        X = torch.Tensor(self.X)
        y = torch.Tensor(self.y)
        self.target.initialise_spectrum(
            model[0].dim, domain=Identity())
        target = self.target.spectrum.T

        def amplitude(model):
            ampl, phase = complex_decompose(model[0].weight)
            return ampl

        overfit_and_plot_progress(
            out_fig=out, model=model, optim=opt, reg=reg_ampl, loss=self.loss,
            max_epoch=self.max_epoch, X=X, y=y, target=target,
            log_interval=self.log_interval, penalise=amplitude, plot=amplitude
        )

        ampl, _ = complex_decompose(model[0].weight)
        target = torch.Tensor(target).squeeze()
        solution = ampl.squeeze()
        score = self.loss(target, solution)
        # This is the score if every guess were exactly 0.1 from the target.
        # (already far better than random chance)
        assert(score < self.max_score)

        # Fewer than 1 in 10 are more than 0.1 from target
        score = ((target - solution).abs() > 0.1).sum().item()
        assert(score < self.dim // 10)
