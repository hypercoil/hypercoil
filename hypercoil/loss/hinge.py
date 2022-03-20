# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Hinge Loss
~~~~~~~~~~
SVM hinge loss.
"""
import torch
from . import ReducingLoss


def hinge_loss(Y_hat, Y):
    return torch.maximum(
        1 - Y * Y_hat,
        torch.zeros_like(Y)
    )


class HingeLoss(ReducingLoss):
    def __init__(self, nu=1, reduction=None, name=None):
        if reduction is None:
            reduction = torch.sum
        super(HingeLoss, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=hinge_loss,
            name=name
        )
