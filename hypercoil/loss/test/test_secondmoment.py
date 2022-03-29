# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for second-moment loss.
"""
import pytest
import torch
from hypercoil.loss.secondmoment import (
    _second_moment,
    SecondMomentCentred,
    SecondMoment
)


class TestSecondMoment:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.n_groups = 3
        self.n_channels = 10
        self.n_observations = 20

    def test_unweighted_variance(self):
        src = torch.zeros(self.n_channels, dtype=torch.long)
        src[(self.n_channels // 2):] = 1
        weight = torch.eye(2)[src].t()
        data = torch.randn(self.n_channels, self.n_observations)

        loss = SecondMoment(standardise=False)
        out = loss(data=data, weight=weight)
        ref = torch.stack((
            data[:(self.n_channels // 2), :].var(-2, unbiased=False),
            data[(self.n_channels // 2):, :].var(-2, unbiased=False)
        ))
        assert torch.allclose(out, ref.mean())

        mu = weight @ data / weight.sum(-1, keepdim=True)
        out = _second_moment(weight, data, mu).squeeze()
        ref = torch.stack((
            data[:(self.n_channels // 2), :].var(-2, unbiased=False),
            data[(self.n_channels // 2):, :].var(-2, unbiased=False)
        ))
        assert torch.allclose(out, ref)

    def test_centre_equivalence(self):
        data = torch.randn(self.n_channels, self.n_observations)
        weight = torch.randn(self.n_groups, self.n_channels)
        mu = weight @ data / weight.sum(-1, keepdim=True)

        loss0 = SecondMoment()
        loss1 = SecondMomentCentred()

        ref = loss0(data=data, weight=weight)
        out = loss1(data=data, weight=weight, mu=mu)
        assert torch.allclose(out, ref)
