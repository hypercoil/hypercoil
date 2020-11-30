# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for operations in the positive semidefinite cone
"""
import numpy as np
import torch
from nilearn.connectome.connectivity_matrices import (
    _form_symmetric, _map_eigenvalues, _geometric_mean
)
from hypernova.functional import (
    tangent_project_spd, cone_project_spd, mean_geom_spd
)


tol = 5e-6
rtol = 5e-5
testf = lambda out, ref: np.allclose(out, ref, atol=tol, rtol=rtol)


A = np.random.rand(10, 10)
AM = np.random.rand(200, 10, 10)
A = A @ A.T
AM = AM @ np.swapaxes(AM, -1, -2)
R = AM.mean(0)
At = torch.Tensor(A)
AMt = torch.Tensor(AM)
Rt = torch.Tensor(R)


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


def test_tangent_project():
    out = tangent_project_spd(At, Rt).numpy()
    ref = nilearn_tangent_project([A], R)
    assert np.allclose(out, ref, atol=1e-1, rtol=1e-1)
    out = tangent_project_spd(AMt, Rt).numpy()
    ref = nilearn_tangent_project(AM, R)
    # Note that this is a very weak condition! This would likely
    # experience major improvement if pytorch develops a proper
    # logm function.
    assert np.allclose(out, ref, atol=2e-1, rtol=2e-1)


def test_cone_project():
    V = nilearn_tangent_project([A], R).squeeze()
    Vt = torch.Tensor(V)
    out = cone_project_spd(Vt, Rt).numpy()
    ref = nilearn_cone_project(V, R)
    assert testf(out, ref)
    out = cone_project_spd(AMt, Rt).numpy()
    ref = np.stack([nilearn_cone_project(AMi, R) for AMi in AM])
    assert testf(out, ref)


def test_proper_inverse():
    Vt = tangent_project_spd(AMt, Rt, recondition=5e-4)
    AM_rec = cone_project_spd(Vt, Rt, recondition=5e-4).numpy()
    assert np.allclose(AM, AM_rec, atol=1e-2, rtol=1e-2)


def test_geometric_mean():
    out = mean_geom_spd(AMt, recondition=1e-6).numpy()
    ref = _geometric_mean([i for i in AM])
    # Another fairly weak condition.
    assert np.allclose(out, ref, atol=1e-2, rtol=1e-2)
