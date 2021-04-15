# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Covariance learning tests
~~~~~~~~~~~~~~~~~~~~~~~~~
Overfitting an autoencoder to test elementary learning capacity for the
covariance module.
"""
import os
import pytest
import torch
import hypernova
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from hypernova.nn import UnaryCovariance
from hypernova.functional import corr
from hypernova.functional.domain import Identity
from hypernova.reg import SymmetricBimodal, SmoothnessPenalty
from hypernova.init.base import DomainInitialiser, uniform_init_
from .synth_aecov import synthesise


class TestCovarianceAutoencoder:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.n = 1000
        os.environ['KMP_DUPLICATE_LIB_OK']='True'
        np.random.seed(0)
        self.X, self.y, self.target = synthesise(n=self.n)
        self.max_epoch = 1000
        self.loss = torch.nn.MSELoss()
        self.log_interval = 25
        self.max_score = np.sqrt(.1 * self.n)

    def test_supervised_cov_autoencoder(self):
        plt.figure(figsize=(9, 18))
        plt.subplot(3, 1, 2)
        color = np.array([0.1, 0.1, 0.1])
        incr = (0.55 - color) / self.max_epoch
        out = '{}/results/test_supervised_cov_autoencoder.svg'.format(
            os.path.dirname(hypernova.__file__))

        torch.manual_seed(0)
        init = DomainInitialiser(
            init=partial(uniform_init_, min=0.4, max=0.6),
            domain=Identity()
        )
        model = UnaryCovariance(
            dim=self.n,
            estimator=corr,
            max_lag=0,
            init=init
        )
        opt = torch.optim.SGD(model.parameters(), lr=1, momentum=0.2)

        reg = [
            SymmetricBimodal(nu=0.01, modes=(0.05, 0.95)),
            SmoothnessPenalty(nu=0.05)
        ]
        X = torch.Tensor(self.X)
        y = torch.Tensor(self.y[-1])
        loss = [float('inf') for _ in range(self.max_epoch)]
        for e in range(self.max_epoch):
            y_hat = model(X).squeeze()
            l = self.loss(y, y_hat) + sum([r(model.weight) for r in reg])
            l.backward()
            opt.step()
            model.zero_grad()
            loss[e] = l.item()
            if e % self.log_interval == 0:
                plt.plot(model.weight.squeeze().detach().numpy(),
                         color=(1 - color))
                color = color + incr * self.log_interval
        plt.plot(model.weight.squeeze().detach().numpy(), color='red')
        plt.gca().set_title('Weight over the course of learning')
        plt.subplot(3, 1, 1)
        plt.plot(loss)
        plt.gca().set_title('Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.subplot(3, 1, 3)
        plt.plot(self.target[:, -1])
        plt.plot(model.weight.squeeze().detach().numpy())
        plt.gca().set_title('Learned and target weights')
        plt.legend(['Target', 'Learned'])
        plt.savefig(out, bbox_inches='tight')

        score = self.loss(
            torch.Tensor(self.target[:, -1]),
            model.weight.squeeze().detach()
        )
        # This is the score if every guess were exactly 0.1 from the target.
        # (already far better than random chance)
        assert(score < self.max_score)
