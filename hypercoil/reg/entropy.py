# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Entropy
~~~~~~~
Loss functions using the entropy of a distribution.
"""
import torch
from functools import partial
from .base import ReducingLoss


def entropy(X, axis=-1):
    """
    Compute the entropy of a categorical distribution.
    """
    eps = torch.finfo(X.dtype).eps
    entropy = -X * torch.log(X + eps)
    return entropy.sum(axis)


def softmax_entropy(X, axis=-1):
    """
    Project logits in the input matrix onto the probability simplex, and then
    compute the entropy of the resulting categorical distribution.
    """
    probs = torch.softmax(X, axis=axis)
    return entropy(probs, axis=axis)


class Entropy(ReducingLoss):
    def __init__(self, nu=1, axis=-1, reduction=None, name=None):
        reduction = reduction or torch.mean
        loss = partial(entropy, axis=axis)
        super(Entropy, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=loss,
            name=name
        )


class SoftmaxEntropy(ReducingLoss):
    def __init__(self, nu=1, axis=-1, reduction=None, name=None):
        reduction = reduction or torch.mean
        loss = partial(softmax_entropy, axis=axis)
        super(SoftmaxEntropy, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=loss,
            name=name
        )
