# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Entropy
~~~~~~~
Regularisations using the entropy of a distribution.
"""
import torch
from functools import partial
from .base import ReducingRegularisation


#TODO: make sure this is fine for half precision, etc.
eps = 1e-8


def entropy(X, axis=-1):
    """
    Compute the entropy of a categorical distribution.
    """
    entropy = -X * torch.log(X + eps)
    return entropy.sum(axis)


def softmax_entropy(X, axis=-1):
    """
    Project logits in the input matrix onto the probability simplex, and then
    compute the entropy of the resulting categorical distribution.
    """
    probs = torch.softmax(X, axis=axis)
    return entropy(probs, axis=axis)


class Entropy(ReducingRegularisation):
    def __init__(self, nu=1, axis=-1, reduction=None):
        reduction = reduction or torch.mean
        reg = partial(entropy, axis=axis)
        super(Entropy, self).__init__(
            nu=nu,
            reduction=reduction,
            reg=reg
        )


class SoftmaxEntropy(ReducingRegularisation):
    def __init__(self, nu=1, axis=-1, reduction=None):
        reduction = reduction or torch.mean
        reg = partial(softmax_entropy, axis=axis)
        super(SoftmaxEntropy, self).__init__(
            nu=nu,
            reduction=reduction,
            reg=reg
        )
