# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Multivariate kurtosis
~~~~~~~~~~~~~~~~~~~~~
Multivariate kurtosis following Mardia, as used by Laumann and colleagues.
"""
import torch
from functools import partial
from .base import ReducingLoss
from ..functional import precision


def multivariate_kurtosis(ts, l2=0):
    prec = precision(ts, l2=l2).unsqueeze(-3)
    ts = ts.transpose(-1, -2).unsqueeze(-2)
    maha = (ts @ prec @ ts.transpose(-1, -2)).squeeze()
    return -(maha ** 2).mean(-1)


class MultivariateKurtosis(ReducingLoss):
    def __init__(self, nu=1, l2=0, reduction=None, name=None):
        reduction = reduction or torch.mean
        loss = partial(multivariate_kurtosis, l2=l2)
        super().__init__(
            nu=nu,
            reduction=reduction,
            loss=loss,
            name=name
        )
