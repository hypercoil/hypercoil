# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Covariance learning tests
~~~~~~~~~~~~~~~~~~~~~~~~~
Overfitting a simple neural network to test elementary learning capacity for
the covariance module.
"""
import os
import pytest
import torch
import hypercoil
import numpy as np
from functools import partial
from hypercoil.nn import UnaryCovariance
from hypercoil.functional import corr
from hypercoil.functional.domain import Identity
from hypercoil.reg import (
    SymmetricBimodal,
    SmoothnessPenalty,
    RegularisationScheme
)
from hypercoil.init.base import DomainInitialiser, uniform_init_
from .synth_netcov import synthesise
from .overfit_plot import overfit_and_plot_progress


class TestCovarianceNetwork:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.n = 1000
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        np.random.seed(0)
        self.X, self.y, self.target = synthesise(n=self.n)
        self.max_epoch = 1000
        self.loss = torch.nn.MSELoss()
        self.log_interval = 25
        self.max_score = np.sqrt(.01 * self.n)

    def test_supervised_cov_network(self):
        out = '{}/results/test_supervised_cov_network.svg'.format(
            os.path.dirname(hypercoil.__file__))

        torch.manual_seed(0)
        init = DomainInitialiser(
            init=partial(uniform_init_, min=0.45, max=0.55),
            domain=Identity()
        )
        model = UnaryCovariance(
            dim=self.n,
            estimator=corr,
            max_lag=0,
            init=init
        )
        opt = torch.optim.SGD(model.parameters(), lr=1, momentum=0.2)

        reg = RegularisationScheme([
            SymmetricBimodal(nu=0.01, modes=(0.05, 0.95)),
            SmoothnessPenalty(nu=0.05)
        ])
        X = torch.Tensor(self.X)
        y = torch.Tensor(self.y[-1])
        target = self.target[:, -1]

        overfit_and_plot_progress(
            out_fig=out, model=model, optim=opt, reg=reg, loss=self.loss,
            max_epoch=self.max_epoch, X=X, y=y, target=target,
            log_interval=self.log_interval
        )

        target = torch.Tensor(self.target[:, -1])
        solution = model.weight.squeeze().detach()

        score = self.loss(target, solution)
        # This is the score if every guess were exactly 0.1 from the target.
        # (already far better than random chance)
        assert(score < self.max_score)

        # Fewer than 1 in 10 are more than 0.1 from target
        score = ((target - solution).abs() > 0.1).sum().item()
        assert(score < self.n // 10)
