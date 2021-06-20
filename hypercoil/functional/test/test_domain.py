# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for preimage/image mappers
"""
import pytest
import numpy as np
import torch
from hypercoil.functional.domainbase import (
    Clip, Normalise, Identity, Linear
)
from hypercoil.functional.domain import (
    Atanh, AmplitudeAtanh, Logit,
    MultiLogit, AmplitudeMultiLogit,
    NullOptionMultiLogit, ANOML
)


class TestDomain:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.A = torch.Tensor([-1.1, -0.5, 0, 0.5, 1, 7])
        self.C = torch.complex(
            torch.Tensor([-1.1, -0.5, 0, 0.5, 1, 7]),
            torch.Tensor([-0.7, -0.2, 1, 1, 0, -5])
        )
        self.AA = torch.Tensor([
            [2, 2, 2, 1, 0],
            [0, 1, 1, 1, 2]
        ])
        ampl_CC = torch.Tensor([
            [2, 2, 2, 1, 0, 0],
            [0, 1, 1, 1, 2, 0]
        ])
        phase_CC = torch.Tensor([
            [-0.7, -0.2, 1, 1, 0, -5],
            [-1.1, -0.5, 0, 0.5, 1, 7]
        ])
        self.CC = ampl_CC * torch.exp(phase_CC * 1j)
        self.Z = torch.rand(5, 3, 4, 4)

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

    def test_multilogit(self):
        dom = MultiLogit(axis=-1)
        out = dom.preimage(self.AA)
        r_in = self.AA
        r_in[r_in < dom.bound[0]] = dom.bound[0]
        r_in[r_in > dom.bound[1]] = dom.bound[1]
        ref = torch.log(r_in)
        assert np.allclose(out, ref)
        out = dom.image(out)
        ref = self.AA / self.AA.sum(-1).view(-1, 1)
        assert np.allclose(out, ref)

    def test_amultilogit(self):
        dom = AmplitudeMultiLogit(axis=0)
        out = dom.preimage(self.CC)
        ampl, phase = torch.abs(self.CC), torch.angle(self.CC)
        ampl[ampl < dom.bound[0]] = dom.bound[0]
        ampl[ampl > dom.bound[1]] = dom.bound[1]
        ref = torch.log(ampl)
        ref = ref * torch.exp(phase * 1j)
        assert np.allclose(out, ref)
        out = dom.image(self.CC)
        ampl, phase = torch.abs(self.CC), torch.angle(self.CC)
        ampl = torch.softmax(ampl, 0)
        ref = ampl * torch.exp(phase * 1j)
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

    def test_noml(self):
        dom = NullOptionMultiLogit(axis=-2)
        out = dom.preimage(self.Z)
        assert out.size() == torch.Size((5, 3, 5, 4))
        assert out.size() == dom.preimage_dim(self.Z)
        assert np.allclose(out[:, :, -1, :], torch.log(dom.bound[0]))
        out = dom.image(out)
        assert torch.all(out.sum(axis=-2) > 0.99)
        out = dom.image(self.Z)
        assert out.size() == torch.Size((5, 3, 3, 4))
