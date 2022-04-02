# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for symmetric matrix maps
"""
import pytest
import numpy as np
import torch
from scipy.linalg import expm, logm, sqrtm, sinm, funm
from hypercoil.functional import (
    symmap, symexp, symlog, symsqrt
)


class TestSymmetricMap:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.tol = 1e-5
        self.rtol = 1e-4
        self.approx = lambda out, ref: np.allclose(
            out, ref, atol=self.tol, rtol=self.rtol)
        np.random.seed(10)
        torch.manual_seed(10)

        A = np.random.rand(10, 10)
        AM = np.random.rand(200, 10, 10)
        AS = np.random.rand(200, 10, 4)
        self.A = A @ A.T
        self.AM = AM @ np.swapaxes(AM, -1, -2)
        self.AS = AS @ np.swapaxes(AS, -1, -2)
        self.At = torch.Tensor(self.A)
        self.AMt = torch.Tensor(self.AM)
        self.ASt = torch.Tensor(self.AS)

        if torch.cuda.is_available():
            self.AtC = self.At.clone().cuda()
            self.AMtC = self.AMt.clone().cuda()

    def test_expm(self):
        out = symexp(self.At).numpy()
        ref = expm(self.A)
        assert self.approx(out, ref)

    def test_logm(self):
        out = symlog(self.At).numpy()
        ref = logm(self.A)
        # Note that this is a very weak condition! This would likely
        # experience major improvement if pytorch develops a proper
        # logm function.
        assert np.allclose(out, ref, atol=1e-2, rtol=1e-2)

    def test_sqrtm(self):
        out = symsqrt(self.At).numpy()
        ref = sqrtm(self.A)
        assert np.allclose(out, ref, atol=1e-3, rtol=1e-3)

    def test_map(self):
        out = symmap(self.At, torch.sin).numpy()
        ref = funm(self.A, np.sin)
        assert self.approx(out, ref)
        ref = sinm(self.A)
        assert self.approx(out, ref)

    def test_map_multidim(self):
        out = symmap(self.AMt, torch.exp).numpy()
        ref = np.stack([expm(AMi) for AMi in self.AM])
        assert self.approx(out, ref)

    def test_singular(self):
        out = symmap(self.ASt, torch.log).numpy()
        #assert np.all(np.logical_or(np.isnan(out), np.isinf(out)))
        out = symmap(self.ASt, torch.log, psi=1e-3).numpy()
        assert np.all(np.logical_not(np.logical_or(
            np.isnan(out), np.isinf(out))))

    @pytest.mark.cuda
    def test_map_cuda(self):
        out = symmap(self.AtC, torch.sin).cpu().numpy()
        ref = funm(self.A, np.sin)
        assert self.approx(out, ref)
        ref = sinm(self.A)
        assert self.approx(out, ref)

    @pytest.mark.cuda
    def test_map_multidim_cuda(self):
        out = symmap(self.AMtC, torch.exp).cpu().numpy()
        ref = np.stack([expm(AMi) for AMi in self.AM])
        assert self.approx(out, ref)
