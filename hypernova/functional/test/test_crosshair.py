# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for crosshair kernel operations
"""
import numpy as np
import torch
from hypernova.functional.crosshair import (
    crosshair_dot,
    crosshair_norm_l2,
    crosshair_norm_l1
)


testf = np.isclose


A = np.random.rand(3, 7, 7)
B = np.random.rand(3, 7, 7)
At = torch.Tensor(A)
Bt = torch.Tensor(B)
index = (1, 3, 2)
indices = [
    (1, 3, 2),
    (1, 3, 0), (1, 3, 1), (1, 3, 3), (1, 3, 4), (1, 3, 5), (1, 3, 6),
    (1, 0, 2), (1, 1, 2), (1, 2, 2), (1, 4, 2), (1, 5, 2), (1, 6, 2)
]


def vector_from_indices(A, indices):
    vec = []
    for i in indices:
        vec += [A[i]]
    return np.array(vec)


def test_crosshair_dot():
    out = crosshair_dot(At, Bt)[index].item()
    ref = vector_from_indices(A, indices) @ vector_from_indices(B, indices)
    assert testf(out, ref)


def test_crosshair_norm_l2():
    out = crosshair_norm_l2(At)[index].item()
    ref = np.linalg.norm(vector_from_indices(A, indices), 2)
    assert testf(out, ref)


def test_crosshair_norm_l1():
    out = crosshair_norm_l1(At)[index].item()
    ref = np.linalg.norm(vector_from_indices(A, indices), 1)
    assert testf(out, ref)
