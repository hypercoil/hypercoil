# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Normed penalty
~~~~~~~~~~~~~~
Generalised module for applying a normed penalty to the weight parameter
of a module.
"""
import torch
from torch.linalg import vector_norm as pnorm
from torch.nn import Module
from functools import partial
from .base import ReducingLoss


def norm_reduction(X, p=2, axis=-1, reduction=None):
    reduction = reduction or torch.mean
    norm = pnorm(X, ord=p, dim=axis)
    return reduction(norm)


class NormedLoss(ReducingLoss):
    def __init__(self, nu, p=2, loss=None, axis=None,
                 reduction=None, name=None):
        reduction = partial(
            norm_reduction,
            p=p,
            axis=axis,
            reduction=reduction
        )
        if loss is None:
            loss = lambda x: x
        super(NormedLoss, self).__init__(
            nu=nu, reduction=reduction, loss=loss, name=name
        )
        self.p = p

    def extra_repr(self):
        return [f'norm=L{self.p}']


class UnilateralNormedLoss(NormedLoss):
    def __init__(self, nu, p=2, loss=None, axis=None,
                 reduction=None, name=None):
        if loss is None:
            loss = lambda x: x
        f = lambda x: torch.maximum(
            loss(x),
            torch.tensor(0, dtype=x.dtype, device=x.device)
        )
        super(UnilateralNormedLoss, self).__init__(
            nu=nu, p=p, loss=f, axis=axis,
            reduction=reduction, name=name
        )
