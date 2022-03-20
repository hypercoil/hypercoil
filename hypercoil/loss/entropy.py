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
    """
    Entropy of a set of categorical distributions.

    Penalising the entropy promotes concentration of weight into a single
    category. Entropy is a concave function. Minimising it without constraint
    affords an unbounded capacity for reducing the loss. This is almost
    certainly undesirable. For this reason, it is recommended that some
    constraint be imposed on the input set when placing a penalty on entropy.
    One possibility is using a multi-logit (softmax) domain mapper to first
    project the input weights onto the probability simplex.

    Parameters
    ----------
    nu : float (default 1)
        Loss function weight multiplier.
    axis : int (default -1)
        Vectors along the specified axis should correspond to the
        probabilities that parameterise a single categorical distribution.
    reduction : callable (default `torch.mean`)
        Map from a tensor of arbitrary dimension to a scalar. The vector of
        entropies computed for each distribution is passed into `reduction`
        to return a scalar.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
    """
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
    """
    Entropy of a set of categorical distributions.

    Penalising the entropy promotes concentration of weight into a single
    category. This is a convenience wrapper that precomposes the entropy with
    a softmax function that first projects the input onto the probability
    simplex. Thus, the input to this loss should contain logits rather than
    probabilities. Use `Entropy` instead if your inputs will already contain
    probabilities.

    Parameters
    ----------
    nu : float (default 1)
        Loss function weight multiplier.
    axis : int (default -1)
        Vectors along the specified axis should correspond to the logits
        that parameterise a single categorical distribution.
    reduction : callable (default `torch.mean`)
        Map from a tensor of arbitrary dimension to a scalar. The vector of
        entropies computed for each distribution is passed into `reduction`
        to return a scalar.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
    """
    def __init__(self, nu=1, axis=-1, reduction=None, name=None):
        reduction = reduction or torch.mean
        loss = partial(softmax_entropy, axis=axis)
        super(SoftmaxEntropy, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=loss,
            name=name
        )