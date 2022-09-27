# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for ephemeral optimisers
"""
import pytest
import torch
from hypercoil.engine.ephemeral import SGDEphemeral, AdamEphemeral


class TestEphemeralOptim:
    def test_ephemeral_sgd(self):
        A = torch.rand(3, 5)
        B0 = torch.rand(6, 3, 3)
        B1 = torch.rand(6, 3, 3)
        A.requires_grad = True
        B0.requires_grad = True
        B1.requires_grad = True

        X = torch.rand(6, 5, 22)

        sgde = SGDEphemeral(params=[A], lr=0.01, momentum=0.9)
        assert len(sgde.param_groups) == 1
        sgde.load_ephemeral(params=[B0])
        assert len(sgde.param_groups) == 2

        B0_before = B0.detach().clone()

        (B0 @ A @ X).sum().backward()
        B0_grad = B0.grad.clone()
        ephemeral_state = sgde.step(return_ephemeral_state=True)
        assert not torch.any(B0 == B0_before)
        assert torch.all(ephemeral_state[B0]['momentum_buffer'] == B0_grad)

        sgde.purge_ephemeral()
        assert len(sgde.param_groups) == 1
        sgde.load_ephemeral(params=[B1])
        assert len(sgde.param_groups) == 2

        B0_before = B0.detach().clone()
        (B1 @ A @ X).sum().backward()
        sgde.step()
        assert torch.all(B0 == B0_before)

    def test_ephemeral_adam(self):
        A = torch.rand(3, 5)
        B0 = torch.rand(6, 3, 3)
        B1 = torch.rand(6, 3, 3)
        A.requires_grad = True
        B0.requires_grad = True
        B1.requires_grad = True

        X = torch.rand(6, 5, 22)

        betas = (0.9, 0.999)
        adame = AdamEphemeral(params=[A], lr=0.01, betas=betas)
        assert len(adame.param_groups) == 1
        adame.load_ephemeral(params=[B0])
        assert len(adame.param_groups) == 2

        B0_before = B0.detach().clone()

        (B0 @ A @ X).sum().backward()
        B0_grad = B0.grad.clone()
        ephemeral_state = adame.step(return_ephemeral_state=True)
        assert not torch.any(B0 == B0_before)
        print(ephemeral_state[B0]['exp_avg_sq'], B0_grad ** 2)
        assert torch.allclose(
            ephemeral_state[B0]['exp_avg'],
            (1 - betas[0]) * B0_grad)
        assert torch.allclose(
            ephemeral_state[B0]['exp_avg_sq'],
            (1 - betas[1]) * (B0_grad ** 2))
