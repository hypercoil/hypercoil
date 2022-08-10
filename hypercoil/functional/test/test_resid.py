# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for residualisation
"""
import numpy as np
from hypercoil.functional.resid import residualise


class TestResidualisation:
    def test_residualisation(self):
        X = np.random.rand(3, 30, 100)
        Y = np.random.rand(3, 1000, 100)
        out = residualise(Y, X)
        assert out.shape == (3, 1000, 100)

        X = np.ones((3, 100, 1))
        Y = np.random.rand(3, 100, 1000)
        out = residualise(Y, X, rowvar=False)
        assert out.shape == (3, 100, 1000)
        assert np.allclose(out.mean(-2), 0, atol=1e-5)
