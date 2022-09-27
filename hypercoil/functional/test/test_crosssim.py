# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for crosshair similarity operations
"""
import pytest
import numpy as np
from hypercoil.functional.crosssim import (
    crosshair_similarity,
    crosshair_cosine_similarity,
    crosshair_l1_similarity,
    crosshair_l2_similarity
)


#TODO: correctness tests for all functions


class TestCrossSim:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.X = np.random.rand(10, 3, 4, 7, 7)
        self.W = np.random.rand(6, 4, 7, 7)
        self.exp_shape = (10, 3, 6, 7, 7)

    def test_crosssim_shape(self):
        out = crosshair_similarity(self.X, self.W)
        assert out.shape == self.exp_shape

    def test_crosssim_cosine_shape(self):
        out = crosshair_cosine_similarity(self.X, self.W)
        assert out.shape == self.exp_shape

    def test_crosssim_l1_shape(self):
        out = crosshair_l1_similarity(self.X, self.W)
        assert out.shape == self.exp_shape

    def test_crosssim_l2_shape(self):
        out = crosshair_l2_similarity(self.X, self.W)
        assert out.shape == self.exp_shape
