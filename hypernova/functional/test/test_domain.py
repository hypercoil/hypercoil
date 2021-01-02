# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for preimage/image mappers
"""
import pytest
import numpy as np
import torch
from hypernova.functional.domain import (
    Clip, Normalise, Identity, Linear, Logit, Atanh, AmplitudeAtanh
)


class TestDomain:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.A = torch.Tensor([-1.1, -0.5, 0, 0.5, 1, 7])
        self.C = torch.complex(
            torch.Tensor([-1.1, -0.5, 0, 0.5, 1, 7]),
            torch.Tensor([-0.7, -0.2, 1, 1, 0, -5])
        )

    def test_clip(self):
        A = torch.Tensor([-0.7, 0.3, 1.2])
        out = Clip().apply(A, bound=[-float('inf'), 1])
        ref = torch.Tensor([-0.7, 0.3, 1])
        assert np.allclose(out, ref)

    def test_norm(self):
        A = torch.Tensor([-0.5, 0, 0.5])
        out = Normalise().apply(A, bound=[0, 1])
        ref = torch.Tensor([0, 0.25, 0.5])
        assert np.allclose(out, ref)

    def test_identity(self):
        dom = Identity()
        out = dom.preimage(self.A)
        ref = self.A
        assert np.allclose(out, ref)
        out = dom.image(self.A)
        ref = self.A
        assert np.allclose(out, ref)

    def test_linear(self):
        dom = Linear(scale=2)
        out = dom.preimage(self.A)
        ref = self.A / 2
        assert np.allclose(out, ref)
        out = dom.image(self.A)
        ref = self.A * 2
        assert np.allclose(out, ref)

    def test_logit(self):
        dom = Logit(scale=2)
        out = dom.preimage(self.A)
        ref = torch.logit(self.A / 2)
        ref[self.A < dom.bound[0]] = dom.limits[0]
        ref[ref < dom.limits[0]] = dom.limits[0]
        ref[self.A > dom.bound[1]] = dom.limits[1]
        ref[ref > dom.limits[1]] = dom.limits[1]
        assert np.allclose(out, ref)
        out = dom.image(self.A)
        ref = torch.sigmoid(self.A) * 2
        assert np.allclose(out, ref)

    def test_atanh(self):
        dom = Atanh(scale=2)
        out = dom.preimage(self.A)
        ref = torch.atanh(self.A / 2)
        ref[self.A < dom.bound[0]] = dom.limits[0]
        ref[ref < dom.limits[0]] = dom.limits[0]
        ref[self.A > dom.bound[1]] = dom.limits[1]
        ref[ref > dom.limits[1]] = dom.limits[1]
        assert np.allclose(out, ref)
        out = dom.image(self.A)
        ref = torch.tanh(self.A) * 2
        assert np.allclose(out, ref)

    def test_aatanh(self):
        dom = AmplitudeAtanh(scale=2)
        out = dom.preimage(self.C)
        ampl, phase = torch.abs(self.C), torch.angle(self.C)
        ref = torch.atanh(ampl / 2)
        ref[ampl < dom.bound[0]] = dom.limits[0]
        ref[ref < dom.limits[0]] = dom.limits[0]
        ref[ampl > dom.bound[1]] = dom.limits[1]
        ref[ref > dom.limits[1]] = dom.limits[1]
        ref = ref * torch.exp(phase * 1j)
        assert np.allclose(out, ref)
        out = dom.image(self.C)
        ampl, phase = torch.abs(self.C), torch.angle(self.C)
        ref = torch.tanh(ampl) * 2
        ref = ref * torch.exp(phase * 1j)
        assert np.allclose(out, ref)
