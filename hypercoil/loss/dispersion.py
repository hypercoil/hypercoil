# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Vector dispersion
~~~~~~~~~~~~~~~~~
Loss functions using mutual separation among a set of vectors.
"""
import torch
from .base import ReducingLoss
from ..functional import sym2vec


def dist(vectors, p=1):
    """
    Unary wrapper for `torch.cdist`.
    """
    return torch.cdist(vectors, vectors, p=p)


class VectorDispersion(ReducingLoss):
    """
    Mutual separation among a set of vectors.

    This can be used as one half of a clustering loss. Such a clustering loss
    promotes mutual separation among centroids (between-cluster separation)
    while also promoting proximity between observations and their closest
    centroids (within-cluster closeness, for instance using a norm loss).

    Parameters
    ----------
    nu : float (default 1)
        Loss function weight multiplier.
    metric : callable (default L1 distance)
        Map from a set of vectors to a distance matrix whose element i, j
        contains some notion of distance between vector i and vector j.
        Note that this object is currently defined with the assumption that
        the `metric` thus passed is commutative.
    reduction : callable (default `torch.mean`)
        Map from a tensor of arbitrary dimension to a scalar. The pairwise
        distance matrix is passed into `reduction` to return a scalar.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
    """
    def __init__(self, nu=1, metric=None, reduction=None, name=None):
        metric = metric or dist
        reduction = reduction or torch.mean
        loss = lambda x: -metric(sym2vec(x))
        super(VectorDispersion, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=loss,
            name=name
        )
