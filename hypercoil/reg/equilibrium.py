# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Equilibrium
~~~~~~~~~~~
Loss functions to favour equal weight across one dimension.
"""
import torch
from functools import partial
from .base import ReducingLoss


def equilibrium(X, axis=-1):
    return X.mean(axis) ** 2


def softmax_equilibrium(X, axis=-1, prob_axis=-2):
    probs = torch.softmax(X, axis=prob_axis)
    return equilibrium(probs, axis=axis)


class Equilibrium(ReducingLoss):
    def __init__(self, nu=1, axis=-1, reduction=None, name=None):
        reduction = reduction or torch.mean
        loss = partial(equilibrium, axis=axis)
        super(Equilibrium, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=loss,
            name=name
        )


class SoftmaxEquilibrium(ReducingLoss):
    def __init__(self, nu=1, axis=-1, prob_axis=-2,
                 reduction=None, name=None):
        reduction = reduction or torch.mean
        loss = partial(softmax_equilibrium, axis=axis, prob_axis=prob_axis)
        super(SoftmaxEquilibrium, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=loss,
            name=name
        )
