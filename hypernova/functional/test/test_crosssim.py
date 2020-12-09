# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for crosshair similarity operations
"""
import torch
from hypernova.functional.crosssim import (
    crosshair_similarity,
    crosshair_cosine_similarity,
    crosshair_l1_similarity,
    crosshair_l2_similarity
)


X = torch.rand(10, 3, 4, 7, 7)
W = torch.rand(6, 4, 7, 7)
exp_shape = torch.Size([10, 3, 6, 7, 7])


def test_crosssim_shape():
    out = crosshair_similarity(X, W)
    assert out.size() == exp_shape


def test_crosssim_cosine_shape():
    out = crosshair_cosine_similarity(X, W)
    assert out.size() == exp_shape


def test_crosssim_l1_shape():
    out = crosshair_l1_similarity(X, W)
    assert out.size() == exp_shape


def test_crosssim_l2_shape():
    out = crosshair_l2_similarity(X, W)
    assert out.size() == exp_shape
