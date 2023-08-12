# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for operations in the positive semidefinite cone
"""
import pytest
import numpy as np
from scipy.linalg import expm, logm
from nilearn.connectome.connectivity_matrices import (
    _form_symmetric, _map_eigenvalues, _geometric_mean
)
from hypercoil.functional import (
    tangent_project_spd, cone_project_spd,
    mean_euc_spd, mean_harm_spd, mean_logeuc_spd, mean_geom_spd
)


def nilearn_tangent_project(input, ref):
    """nilearn's method for projecting into tangent space"""
    vals, vecs = np.linalg.eigh(ref)
    inv_sqrt = _form_symmetric(np.sqrt, 1. / vals, vecs)
    whitened_matrices = [inv_sqrt.dot(matrix).dot(inv_sqrt)
                         for matrix in input]
    return np.stack([_map_eigenvalues(np.log, w_mat)
                    for w_mat in whitened_matrices])


def nilearn_cone_project(input, ref):
    """nilearn's method for projecting into PSD cone"""
    rvals, rvecs = np.linalg.eigh(ref)
    ivals, ivecs = np.linalg.eigh(input)
    sqrt = _form_symmetric(np.sqrt, rvals, rvecs)
    # Move along the geodesic
    return sqrt.dot(_form_symmetric(np.exp, ivals, ivecs)).dot(sqrt)


class TestSemidefinite:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.tol = 5e-6
        self.rtol = 5e-5
        self.approx = lambda out, ref: np.allclose(
            out, ref, atol=self.tol, rtol=self.tol)
        np.random.seed(10)

        A = np.random.rand(10, 10)
        AM = np.random.rand(200, 10, 10)
        self.A = A @ A.T
        self.AM = AM @ np.swapaxes(AM, -1, -2)
        self.R = self.AM.mean(0)

    def test_tangent_project(self):
        out = tangent_project_spd(self.A, self.R)
        ref = nilearn_tangent_project([self.A], self.R)
        assert np.allclose(out, ref, atol=1e-1, rtol=1e-1)
        out = tangent_project_spd(self.AM, self.R)
        ref = nilearn_tangent_project(self.AM, self.R)
        # Note that this is a very weak condition! This would likely
        # experience major improvement if pytorch develops a proper
        # logm function.
        diff = out - ref
        assert np.sum(diff > 0.1) < 100

    def test_cone_project(self):
        V = nilearn_tangent_project([self.A], self.R).squeeze()
        out = cone_project_spd(V, self.R)
        ref = nilearn_cone_project(V, self.R)
        assert np.allclose(out, ref, atol=1e-2, rtol=1e-2)
        out = cone_project_spd(self.AM, self.R)
        ref = np.stack([nilearn_cone_project(AMi, self.R) for AMi in self.AM])
        assert np.allclose(out, ref, atol=1e-2, rtol=1e-2)

    def test_means(self):
        out = mean_euc_spd(self.AM, axis=0)
        ref = self.AM.mean(0)
        assert np.allclose(out, ref)
        out = mean_harm_spd(self.AM, axis=0)
        ref = np.linalg.inv(np.linalg.inv(self.AM).mean(0))
        assert np.allclose(out, ref, atol=1e-2, rtol=1e-2)
        out = mean_logeuc_spd(self.AM, axis=0)
        ref = expm(np.stack([logm(m) for m in self.AM]).mean(0))
        assert np.allclose(out, ref, atol=1e-2, rtol=1e-2)


    def test_geometric_mean(self):
        out = mean_geom_spd(self.AM, recondition=1e-3)
        ref = _geometric_mean([i for i in self.AM])
        # Another fairly weak condition.
        assert np.abs(out - ref).max() < .1
