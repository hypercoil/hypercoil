# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for Dirichlet initialisation
"""
import pytest
import torch
from hypercoil.init.dirichlet import DirichletInit
from hypercoil.functional.domain import Identity


class TestDirichletInit:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.X = torch.Tensor(2, 3, 4, 5)
        self.X.requires_grad = True
        self.conc = torch.Tensor([0.5] * 3)

    def test_dirichlet_init_softmax(self):
        init = DirichletInit(n_classes=3, concentration=self.conc, axis=-3)
        init(self.X)
        assert torch.allclose(
            init.domain.image(self.X).sum(-3),
            torch.tensor(1.0)
        )

    def test_dirichlet_init(self):
        init = DirichletInit(
            n_classes=3, concentration=self.conc,
            axis=-3, domain=Identity()
        )
        init(self.X)
        assert torch.allclose(self.X.sum(-3), torch.tensor(1.0))

