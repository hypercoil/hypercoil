# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for differentiable terminals.
"""
import pytest
import torch
from hypercoil.engine.terminal import ReactiveTerminal
from hypercoil.loss.secondmoment import SecondMoment


class TestTerminals:

    def test_reactive_terminal_gradient(self):
        torch.manual_seed(0)
        n_groups = 3
        n_channels = 10
        n_observations = 20
        data = torch.randn(n_channels, n_observations)
        weight = torch.randn(n_groups, n_channels)
        data.requires_grad = True
        weight.requires_grad = True

        loss = SecondMoment(standardise=False)
        terminal = ReactiveTerminal(
            loss=SecondMoment(standardise=False),
            slice_target='data',
            slice_axis=-1,
            max_slice=1
        )

        Y0 = loss(data=data, weight=weight)
        #TODO: make this an argument object, probably. Not that it matters.
        Y1 = terminal(arg={'data': data, 'weight': weight})
        assert torch.isclose(Y0, Y1)

        g1 = weight.grad.clone()
        weight.grad.zero_()
        Y0.backward()
        g0 = weight.grad.clone()
        assert torch.allclose(g0, g1)
