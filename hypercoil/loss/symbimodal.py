# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Symmetric bimodal
~~~~~~~~~~~~~~~~~
Symmetric bimodal regularisations using absolute value.
"""
from functools import partial
from .base import ReducingLoss
from .norm import NormedLoss


def symmetric_bimodal_distance(weight, modes=(0, 1)):
    mean = sum(modes) / 2
    step = max(modes) - mean
    return ((weight - mean).abs() - step).abs()


class SymmetricBimodalNorm(NormedLoss):
    """
    Loss based on the norm of the minimum distance from either of two modes.

    Penalising this quantity can be used to concentrate weights at two modes,
    for instance 0 and 1 or -1 and 1.

    Parameters
    ----------
    nu : float
        Loss function weight multiplier.
    modes : tuple(float, float) (default (0, 1))
        Modes of the loss.
    norm : float (default 2)
        Norm order. p=1 corresponds to the Manhattan L1 norm, p=2 corresponds
        to the Euclidean L2 norm, etc.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
    """
    def __init__(self, nu=1, modes=(0, 1), norm=2, name=None):
        precompose = partial(
            symmetric_bimodal_distance,
            modes=modes
        )
        super(SymmetricBimodalNorm, self).__init__(
            nu=nu, p=norm, precompose=precompose, name=name)


class SymmetricBimodal(ReducingLoss):
    """
    Loss based on the norm of the minimum distance from either of two modes.

    Penalising this quantity can be used to concentrate weights at two modes,
    for instance 0 and 1 or -1 and 1.

    Parameters
    ----------
    nu : float
        Loss function weight multiplier.
    modes : tuple(float, float) (default (0, 1))
        Modes of the loss.
    reduction : callable (default `torch.mean`)
        Map from a tensor of arbitrary dimension to a scalar. The tensor of
        minimum distances from either mode is passed into `reduction` to
        return a scalar..
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
    """
    def __init__(self, nu=1, modes=(0, 1), reduction=None, name=None):
        reduction = None or torch.mean
        loss = partial(
            symmetric_bimodal_distance,
            modes=modes
        )
        super(SymmetricBimodal, self).__init__(
            nu=nu, reduction=reduction, loss=loss, name=name)
