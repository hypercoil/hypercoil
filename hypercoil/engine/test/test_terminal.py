# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for differentiable terminals.
"""
import pytest
import torch
from functools import partial
from hypercoil.engine.terminal import (
    ReactiveTerminal,
    ReactiveMultiTerminal
)
from hypercoil.loss import (
    SecondMoment,
    SecondMomentCentred,
    Compactness
)


class TestTerminals:

    def test_reactive_terminal_gradient_secondmoment(self):
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

    def test_reactive_terminal_gradient_cmass(self):
        torch.manual_seed(0)
        n_groups = 4
        n_channels = 10
        n_dims = 3
        coor = torch.randn(n_dims, n_channels)
        weight = torch.randn(n_groups, n_channels)
        coor.requires_grad = True
        weight.requires_grad = True

        loss = Compactness(coor=coor)
        terminal = ReactiveTerminal(
            loss=Compactness(coor=coor),
            slice_target='X',
            slice_axis=-2,
            max_slice=1
        )

        Y0 = loss(X=weight)
        #TODO: make this an argument object, probably. Not that it matters.
        Y1 = terminal(arg={'X': weight})
        assert torch.isclose(Y0, Y1)

        g1 = weight.grad.clone()
        weight.grad.zero_()
        Y0.backward()
        g0 = weight.grad.clone()
        assert torch.allclose(g0, g1)

    def test_reactive_multiterminal_gradient(self):
        torch.manual_seed(0)
        n_groups = 3
        n_channels = 10
        n_observations = 20
        data = torch.randn(n_channels, n_observations)
        weight = torch.randn(n_groups, n_channels)
        mu = torch.randn(n_groups, n_observations)
        data.requires_grad = True
        weight.requires_grad = True

        loss = SecondMomentCentred()
        terminal = ReactiveMultiTerminal(
            loss=SecondMomentCentred(),
            slice_instructions={'data': -1, 'mu': -1},
            max_slice=1
        )

        Y0 = loss(data=data, weight=weight, mu=mu)
        #TODO: make this an argument object, probably. Not that it matters.
        Y1 = terminal(arg={'data': data, 'weight': weight, 'mu': mu})
        assert torch.isclose(Y0, Y1)

        g1 = weight.grad.clone()
        weight.grad.zero_()
        Y0.backward()
        g0 = weight.grad.clone()
        assert torch.allclose(g0, g1)

    def test_reactive_terminal_pretransform(self):
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
            max_slice=1,
            pretransforms={
                'weight': partial(torch.softmax, axis=-2),
                'data': partial(torch.softmax, axis=-2)
            }
        )

        Y0 = loss(
            data=torch.softmax(data, axis=-2),
            weight=torch.softmax(weight, axis=-2)
        )
        #TODO: make this an argument object, probably. Not that it matters.
        Y1 = terminal(arg={'data': data, 'weight': weight})
        assert torch.isclose(Y0, Y1)

        g1 = weight.grad.clone()
        weight.grad.zero_()
        Y0.backward()
        g0 = weight.grad.clone()
        assert torch.allclose(g0, g1)

        mu = torch.randn(n_groups, n_observations)
        loss = SecondMomentCentred()
        terminal = ReactiveMultiTerminal(
            loss=SecondMomentCentred(),
            slice_instructions={'data': -1, 'mu': -1},
            max_slice=1,
            pretransforms={
                'weight': partial(torch.softmax, axis=-2),
                'data': partial(torch.softmax, axis=-2)
            }
        )

        Y0 = loss(
            data=torch.softmax(data, axis=-2),
            weight=torch.softmax(weight, axis=-2),
            mu=mu
        )
        #TODO: make this an argument object, probably. Not that it matters.
        Y1 = terminal(arg={'data': data, 'weight': weight, 'mu': mu})
        assert torch.isclose(Y0, Y1)

        g1 = weight.grad.clone()
        weight.grad.zero_()
        Y0.backward()
        g0 = weight.grad.clone()
        assert torch.allclose(g0, g1, atol=1e-3)

    def test_reactive_terminal_pretransform_mask(self):
        torch.manual_seed(0)
        n_groups = 3
        n_channels = 10
        n_observations = 20
        mask = torch.tensor(
            [1 for _ in range(10)] + [0 for _ in range(10)],
            dtype=torch.bool
        )
        data = torch.randn(n_channels, n_observations)
        weight = torch.randn(n_groups, n_channels)
        weight.requires_grad = True

        loss = SecondMoment(standardise=False)
        terminal = ReactiveTerminal(
            loss=SecondMoment(standardise=False),
            slice_target='data',
            slice_axis=-1,
            max_slice=1,
            pretransforms={
                'weight': partial(torch.softmax, axis=-2),
                'data': partial(torch.softmax, axis=-2)
            }
        )

        Y0 = loss(
            data=torch.softmax(data[:, :10], axis=-2),
            weight=torch.softmax(weight, axis=-2)
        )
        #TODO: make this an argument object, probably. Not that it matters.
        Y1 = terminal(arg={'data': data, 'weight': weight}, axis_mask=mask)
        assert torch.isclose(Y0, Y1)
