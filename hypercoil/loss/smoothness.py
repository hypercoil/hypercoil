# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Penalise backwards differences to favour smoothness.
"""
import torch
from functools import partial
from .norm import NormedLoss


class SmoothnessPenalty(NormedLoss):
    """
    Loss based on the norm of differences.

    Penalising this quantity can minimise transitions or promote smoother
    transitions between adjacent entries in a weight tensor.

    Parameters
    ----------
    nu : float
        Loss function weight multiplier.
    axis : int (default -1)
        Axis defining the slice of the input tensor over which the norm of
        differences is computed.is computed.
    append, prepend (default None)
        Arguments to `torch.diff`. Values to append or prepend to the input
        along the specified axis before computing the difference.
    norm : float (default 2)
        Norm order. p=1 corresponds to the Manhattan L1 norm, p=2 corresponds
        to the Euclidean L2 norm, etc.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
    """
    def __init__(self, nu=1, axis=-1, prepend=None,
                 append=None, norm=2, name=None):
        precompose = partial(
            torch.diff, dim=axis, prepend=prepend, append=append
        )
        super(SmoothnessPenalty, self).__init__(
            nu=nu,
            p=norm,
            precompose=precompose,
            axis=axis,
            name=name
        )
