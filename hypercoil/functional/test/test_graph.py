# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for graph and network measures
"""
import pytest
import numpy as np
import torch
from hypercoil.functional import (
    modularity_matrix,
    relaxed_modularity
)
from communities.utilities import (
    modularity_matrix as modularity_matrix_ref,
    modularity as modularity_ref
)


class TestGraph:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.tol = 5e-7
        self.approx = lambda out, ref: np.allclose(out, ref, atol=self.tol)

        self.X = np.random.rand(3, 20, 20)
        self.X += self.X.swapaxes(-1, -2)
        self.aff = np.random.randint(0, 4, 20)
        self.comms = [np.where(self.aff==c)[0] for c in np.unique(self.aff)]
        self.C = np.eye(4)[self.aff]
        self.Xt = torch.Tensor(self.X)
        self.Ct = torch.Tensor(self.C)
        self.Lt = torch.rand(4, 4)

    def test_modularity_matrix(self):
        out = modularity_matrix(self.Xt, normalise=True)
        ref = np.stack([modularity_matrix_ref(x) for x in self.X])
        assert self.approx(out, ref)

    def test_modularity(self):
        out = relaxed_modularity(self.Xt, self.Ct,
                                 exclude_diag=True,
                                 directed=False)
        ref = np.stack(
            [modularity_ref(modularity_matrix_ref(x), self.comms)
             for x in self.X])
        assert self.approx(out, ref)

    def test_nonassociative_block(self):
        out = relaxed_modularity(self.Xt, self.Ct,
                                 L=self.Lt, exclude_diag=True) / 2
