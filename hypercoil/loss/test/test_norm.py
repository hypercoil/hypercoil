# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for normed losses
"""
import pytest
import torch
from hypercoil.loss import (
    NormedLoss,
    UnilateralNormedLoss,
    ConstraintViolation
)


class TestNormedLosses:

    def test_normed_losses(self):
        X = torch.tensor([
            [-1, 2, 0, 2, 1],
            [0, 1, 0, 0, -1],
            [3, -1, 0, -2, 0]
        ], dtype=torch.float)

        L0 = NormedLoss(nu=1, p=0)
        assert L0(X) == 9
        L0 = NormedLoss(nu=1, p=0, axis=0)
        assert L0(X) == (X != 0).sum() / 5
        L0 = NormedLoss(nu=1, p=0, axis=-1)
        assert L0(X) == (X != 0).sum() / 3

        L1 = NormedLoss(nu=1, p=1)
        assert L1(X) == 14
        L1 = NormedLoss(nu=1, p=1, axis=0)
        assert L1(X) == X.abs().sum() / 5
        L1 = NormedLoss(nu=1, p=1, axis=-1)
        assert L1(X) == X.abs().sum() / 3

        L2 = NormedLoss(nu=1, p=2)
        assert L2(X) == (X ** 2).sum().sqrt()
        L2 = NormedLoss(nu=1, p=2, axis=0)
        assert L2(X) == (X ** 2).sum(0).sqrt().mean()
        L2 = NormedLoss(nu=1, p=2, axis=-1)
        assert L2(X) == (X ** 2).sum(-1).sqrt().mean()

        uL0 = UnilateralNormedLoss(nu=1, p=0)
        assert uL0(X) == 5
        assert uL0(-X) == 4
        uL1 = UnilateralNormedLoss(nu=1, p=1)
        assert uL1(X) == 9
        assert uL1(-X) == 5
        uL2 = UnilateralNormedLoss(nu=1, p=2)
        assert uL2(X) == torch.tensor(19).sqrt()
        assert uL2(-X) == torch.tensor(7).sqrt()

    def test_constraint_violation(self):
        X = torch.tensor([
            [1, -1],
            [-2, 1],
            [-1, 2],
            [0, 0]
        ], dtype=torch.float)

        constraints = [
            lambda x: x
        ]
        V = ConstraintViolation(nu=1, constraints=constraints, p=1)
        U = UnilateralNormedLoss(nu=1, p=1)
        assert V(X) == 4
        assert V(X) == U(X)

        constraints = [
            lambda x: x @ torch.tensor([[1.], [1.]])
        ]
        V = ConstraintViolation(nu=1, constraints=constraints, p=1)
        assert V(X) == 1

        constraints = [
            lambda x: x @ torch.tensor([[0.], [1.]]),
            lambda x: x @ torch.tensor([[1.], [0.]]),
        ]
        V = ConstraintViolation(nu=1, constraints=constraints, p=0)
        assert V(X) == 3

        constraints = [
            lambda x: torch.tensor([1., 1., 1., 1.]) @ x,
        ]
        V = ConstraintViolation(nu=1, constraints=constraints, p=1)
        assert V(X) == 2

        constraints = [
            lambda x: x @ torch.tensor([[-1.], [1.]]),
        ]
        V = ConstraintViolation(nu=1, constraints=constraints, p=1)
        assert V(X) == 6
