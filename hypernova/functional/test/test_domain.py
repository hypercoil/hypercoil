# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for preimage/image mappers
"""
import numpy as np
import torch
from hypernova.functional.domain import (
    Clip, Normalise, Identity, Linear, Logit, Atanh, AmplitudeAtanh
)


A = torch.Tensor([-1.1, -0.5, 0, 0.5, 1, 7])
C = torch.complex(
    torch.Tensor([-1.1, -0.5, 0, 0.5, 1, 7]),
    torch.Tensor([-0.7, -0.2, 1, 1, 0, -5])
)


def test_clip():
    A = torch.Tensor([-0.7, 0.3, 1.2])
    out = Clip().apply(A, bound=[-float('inf'), 1])
    ref = torch.Tensor([-0.7, 0.3, 1])
    assert np.allclose(out, ref)


def test_norm():
    A = torch.Tensor([-0.5, 0, 0.5])
    out = Normalise().apply(A, bound=[0, 1])
    ref = torch.Tensor([0, 0.25, 0.5])
    assert np.allclose(out, ref)


def test_identity():
    dom = Identity()
    out = dom.preimage(A)
    ref = A
    assert np.allclose(out, ref)
    out = dom.image(A)
    ref = A
    assert np.allclose(out, ref)


def test_linear():
    dom = Linear(scale=2)
    out = dom.preimage(A)
    ref = A / 2
    assert np.allclose(out, ref)
    out = dom.image(A)
    ref = A * 2
    assert np.allclose(out, ref)


def test_logit():
    dom = Logit(scale=2)
    out = dom.preimage(A)
    ref = torch.logit(A / 2)
    ref[A < dom.bound[0]] = dom.limits[0]
    ref[ref < dom.limits[0]] = dom.limits[0]
    ref[A > dom.bound[1]] = dom.limits[1]
    ref[ref > dom.limits[1]] = dom.limits[1]
    assert np.allclose(out, ref)
    out = dom.image(A)
    ref = torch.sigmoid(A) * 2
    assert np.allclose(out, ref)


def test_atanh():
    dom = Atanh(scale=2)
    out = dom.preimage(A)
    ref = torch.atanh(A / 2)
    ref[A < dom.bound[0]] = dom.limits[0]
    ref[ref < dom.limits[0]] = dom.limits[0]
    ref[A > dom.bound[1]] = dom.limits[1]
    ref[ref > dom.limits[1]] = dom.limits[1]
    assert np.allclose(out, ref)
    out = dom.image(A)
    ref = torch.tanh(A) * 2
    assert np.allclose(out, ref)


def test_aatanh():
    dom = AmplitudeAtanh(scale=2)
    out = dom.preimage(C)
    ampl, phase = torch.abs(C), torch.angle(C)
    ref = torch.atanh(ampl / 2)
    ref[ampl < dom.bound[0]] = dom.limits[0]
    ref[ref < dom.limits[0]] = dom.limits[0]
    ref[ampl > dom.bound[1]] = dom.limits[1]
    ref[ref > dom.limits[1]] = dom.limits[1]
    ref = ref * torch.exp(phase * 1j)
    assert np.allclose(out, ref)
    out = dom.image(C)
    ampl, phase = torch.abs(C), torch.angle(C)
    ref = torch.tanh(ampl) * 2
    ref = ref * torch.exp(phase * 1j)
    assert np.allclose(out, ref)
