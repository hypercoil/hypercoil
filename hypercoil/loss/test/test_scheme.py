# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for loss schemes.
"""
import pytest
import torch
from hypercoil.engine.terminal import (
    ReactiveTerminal
)
from hypercoil.loss import (
    LossScheme,
    LossApply,
    LossArgument,
    Entropy,
    Equilibrium,
    Compactness
)


class TestLossScheme:

    def test_spatial_loss_equivalence(self):
        torch.manual_seed(0)
        n_groups = 4
        n_channels = 10
        n_dims = 3
        coor = torch.randn(n_dims, n_channels)
        weight = torch.randn(n_groups, n_channels)
        weight = torch.softmax(weight, axis=-2)
        coor.requires_grad = True
        weight.requires_grad = True

        entropy = Entropy(nu=0.2)
        equilibrium = Equilibrium(nu=20)
        compactness = ReactiveTerminal(
            Compactness(nu=2, coor=coor),
            slice_target='X',
            slice_axis=-2,
            max_slice=2
        )

        overall_scheme = LossScheme((
            LossScheme(
                (entropy, equilibrium),
                apply=lambda arg: arg.X
            ),
            compactness
        ))

        ref = entropy(weight) + equilibrium(weight)
        ref.backward()
        _ = compactness({'X': weight})

        g_ref = weight.grad.clone()
        weight.grad.zero_()

        arg = LossArgument(X=weight)
        out = overall_scheme(arg)
        out.backward()
        g_out = weight.grad.clone()

        assert torch.isclose(out, ref)
        assert torch.allclose(g_out, g_ref)
