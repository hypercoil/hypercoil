# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Autoencoder tests
~~~~~~~~~~~~~~~~~
Overfitting autoencoders to test elementary learning capacity.
"""
import pytest
import torch
import hypernova
import numpy as np
from scipy.fft import rfft, irfft
from scipy.special import expit


class TestAutoencoders:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.max_epoch = 1000
        self.optim = torch.optim.SGD
        self.optim_params = {
            'lr': 1e-2,
            'momentum': 0.5
        }
        self.loss = lambda y_hat, y: ((y_hat - y) ** 2).sum()

        self.p = 7
        self.n = 100

        _filter_specs = [hypernova.init.IIRFilterSpec(
            0.5, btype='lowpass', clamps = [{0: 0}]
        )]
        self.filter_input = np.random.randn(self.p, self.n)
        self.filter_weight = expit(np.linspace(3, -6, 51))
        self.filter_weight[0] = 0
        _target = irfft(
            rfft(self.filter_input) * self.filter_weight)
        self.filter_target = irfft(
            rfft(np.flip(_target)) * self.filter_weight)
        self.filter_layer = hypernova.nn.FrequencyDomainFilter(
            filter_specs=_filter_specs,
            dim=51
        )

        self.cov_input = self.filter_target
        self.cov_weight = np.random.rand(self.n)
        self.cov_target = np.cov(self.cov_input, aweights=self.cov_weight)
        self.cov_layer = hypernova.nn.UnaryCovariance(
            dim=self.n,
            estimator=hypernova.functional.cov,
        )

    def _overfit_autoencoder(self, model, input, target):
        optim = self.optim(model.parameters(), **self.optim_params)
        for _ in range(self.max_epoch):
            output = model(input)
            loss = self.loss(output, target)
            loss.backward()
            optim.step()

    def test_filter_autoencoder(self):
        self._overfit_autoencoder(
            self.filter_layer,
            torch.from_numpy(self.filter_input).type(torch.complex64),
            torch.from_numpy(self.filter_target))
        print(self.filter_layer.weight)
        assert False

    def test_cov_autoencoder(self):
        print(self.cov_layer.weight)
        print(np.diag(self.cov_weight))
        self._overfit_autoencoder(
            self.cov_layer,
            torch.from_numpy(self.cov_input).type(torch.float),
            torch.from_numpy(self.cov_target).type(torch.float)
        )
        print(self.cov_layer.weight)
        assert False
