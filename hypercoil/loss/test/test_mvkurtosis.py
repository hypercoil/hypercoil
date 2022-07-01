# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for multivariate kurtosis
"""
import pytest
import torch
from hypercoil.loss.mvkurtosis import MultivariateKurtosis


class TestMultivariateKurtosis:

    def test_expected_value(self):
        mvk = MultivariateKurtosis()
        mvks = MultivariateKurtosis(dimensional_scaling=True)

        dims = (5, 10, 20, 50, 100)
        for d in dims:
            ref = -torch.tensor(d * (d + 2), dtype=torch.float)
            ts = torch.randn(10, d, 2000)
            out = mvk(ts)
            assert torch.isclose(out, ref, rtol=0.05)
            out = mvks(ts)
            assert torch.isclose(out, torch.tensor(-1.), atol=0.01)
