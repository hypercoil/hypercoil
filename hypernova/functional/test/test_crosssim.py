# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for crosshair similarity operations
"""
import pytest
import torch
from hypernova.functional.crosssim import (
    crosshair_similarity,
    crosshair_cosine_similarity,
    crosshair_l1_similarity,
    crosshair_l2_similarity
)


class TestCrossSim:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.X = torch.rand(10, 3, 4, 7, 7)
        self.W = torch.rand(6, 4, 7, 7)
        self.exp_shape = torch.Size([10, 3, 6, 7, 7])

    def test_crosssim_shape(self):
        out = crosshair_similarity(self.X, self.W)
        assert out.size() == self.exp_shape

    def test_crosssim_cosine_shape(self):
        out = crosshair_cosine_similarity(self.X, self.W)
        assert out.size() == self.exp_shape

    def test_crosssim_l1_shape(self):
        out = crosshair_l1_similarity(self.X, self.W)
        assert out.size() == self.exp_shape

    def test_crosssim_l2_shape(self):
        out = crosshair_l2_similarity(self.X, self.W)
        assert out.size() == self.exp_shape
