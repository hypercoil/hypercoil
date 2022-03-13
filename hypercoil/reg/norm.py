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
from .base import ReducingRegularisation


def norm_reduction(X, p=2, axis=-1, reduction=None):
    reduction = reduction or torch.mean
    norm = pnorm(X, ord=p, dim=axis)
    return reduction(norm)


class NormedRegularisation(ReducingRegularisation):
    def __init__(self, nu, p=2, reg=None, axis=None, reduction=None):
        reduction = reduction or torch.mean
        reduction = partial(
            norm_reduction,
            p=p,
            axis=axis,
            reduction=reduction
        )
        if reg is None:
            reg = lambda x: x
        super(NormedRegularisation, self).__init__(
            nu=nu, reduction=reduction, reg=reg
        )
        self.p = p

    def extra_repr(self):
        return f'norm=L{self.p}'


class UnilateralNormedRegularisation(NormedRegularisation):
    def __init__(self, nu, p=2, reg=None):
        if reg is None:
            reg = lambda x: x
        r = lambda x: torch.maximum(reg(x), torch.tensor(0))
        super(UnilateralNormedRegularisation, self).__init__(
            nu=nu, p=p, reg=r
        )
        self.p = p
