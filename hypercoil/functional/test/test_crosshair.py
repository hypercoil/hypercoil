# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for crosshair kernel operations
"""
import pytest
import numpy as np
import torch
from hypercoil.functional.crosshair import (
    crosshair_dot,
    crosshair_norm_l2,
    crosshair_norm_l1
)


#TODO: Unit tests still needed for:
# - Generalised crosshair dot in >2 dims


def vector_from_indices(A, indices):
    vec = []
    for i in indices:
        vec += [A[i]]
    return np.array(vec)


class TestCrosshair:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.approx = np.isclose
        self.A = np.random.rand(3, 7, 7)
        self.B = np.random.rand(3, 7, 7)
        self.At = torch.Tensor(self.A)
        self.Bt = torch.Tensor(self.B)
        self.index = (1, 3, 2)
        self.indices = [
            (1, 3, 2),
            (1, 3, 0), (1, 3, 1), (1, 3, 3), (1, 3, 4), (1, 3, 5), (1, 3, 6),
            (1, 0, 2), (1, 1, 2), (1, 2, 2), (1, 4, 2), (1, 5, 2), (1, 6, 2)
        ]
        if torch.cuda.is_available():
            self.AtC = self.At.clone().cuda()
            self.BtC = self.Bt.clone().cuda()

    def test_crosshair_dot(self):
        out = crosshair_dot(self.At, self.Bt)[self.index].item()
        ref = vector_from_indices(self.A, self.indices
            ) @ vector_from_indices(self.B, self.indices)
        assert self.approx(out, ref)

    def test_crosshair_norm_l2(self):
        out = crosshair_norm_l2(self.At)[self.index].item()
        ref = np.linalg.norm(vector_from_indices(self.A, self.indices), 2)
        assert self.approx(out, ref)

    def test_crosshair_norm_l1(self):
        out = crosshair_norm_l1(self.At)[self.index].item()
        ref = np.linalg.norm(vector_from_indices(self.A, self.indices), 1)
        assert self.approx(out, ref)

    @pytest.mark.cuda
    def test_crosshair_dot_cuda(self):
        out = crosshair_dot(self.AtC, self.BtC)[self.index].cpu().item()
        ref = vector_from_indices(self.A, self.indices
            ) @ vector_from_indices(self.B, self.indices)
        assert self.approx(out, ref)

    @pytest.mark.cuda
    def test_crosshair_norm_l2_cuda(self):
        out = crosshair_norm_l2(self.AtC)[self.index].cpu().item()
        ref = np.linalg.norm(vector_from_indices(self.A, self.indices), 2)
        assert self.approx(out, ref)

    @pytest.mark.cuda
    def test_crosshair_norm_l1_cuda(self):
        out = crosshair_norm_l1(self.AtC)[self.index].cpu().item()
        ref = np.linalg.norm(vector_from_indices(self.A, self.indices), 1)
        assert self.approx(out, ref)
