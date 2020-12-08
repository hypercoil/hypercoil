# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for graph and network measures
"""
import numpy as np
import torch
from hypernova.functional import (
    modularity_matrix
)
from communities.utilities import (
    modularity_matrix as modularity_matrix_ref,
    modularity as modularity_ref
)


tol = 5e-7
testf = lambda out, ref: np.allclose(out, ref, atol=tol)

X = np.random.rand(3, 7, 7)
X += X.swapaxes(-1, -2)
Xt = torch.Tensor(X)


def test_modularity_matrix():
    out = modularity_matrix(Xt, normalise=True)
    ref = np.stack([modularity_matrix_ref(x) for x in X])
    assert testf(out, ref)
