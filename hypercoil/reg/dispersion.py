# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Vector dispersion
~~~~~~~~~~~~~~~~~
Regularisations using mutual separation among a set of vectors.
"""
import torch
from .base import ReducingRegularisation
from ..functional import sym2vec


def dist(vectors, p=1):
    return torch.cdist(vectors, vectors, p=p)


class VectorDispersion(ReducingRegularisation):
    def __init__(self, nu=1, metric=None, reduction=None):
        metric = metric or dist
        reduction = reduction or torch.mean
        reg = lambda x: -sym2vec(metric(x))
        super(VectorDispersion, self).__init__(
            nu=nu,
            reduction=reduction,
            reg=reg
        )
