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
        self.max_epoch = 10
        self.optim = torch.optim.SGD
        self.optim_params = {
            'lr': 5e-2,
            'momentum': 0.5
        }
        self.loss = lambda y_hat, y: ((y_hat - y) ** 2).sum()

        _filter_specs = [hypernova.init.IIRFilterSpec(
            0.5, btype='lowpass', clamps = [{0: 0}]
        )]
        self.filter_input = np.random.randn(100)
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

    def _overfit_autoencoder(self, model, input, target):
        optim = self.optim(model.parameters(), **self.optim_params)
        for _ in range(self.max_epoch):
            output = model(input)
            loss = self.loss(output, target)
            loss.backward()
            optim.step()
            model.zero_grad()

    def test_filter_autoencoder(self):
        self._overfit_autoencoder(
            self.filter_layer,
            torch.from_numpy(self.filter_input).type(torch.complex64),
            torch.from_numpy(self.filter_target))
        print(self.filter_layer.weight)
        assert False
