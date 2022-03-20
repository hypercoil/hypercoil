# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Second Moment
~~~~~~~~~~~~~
Regularise the second moment, e.g. to favour regions whose time series are
homogeneous across space.
"""
from torch import mean
from .base import ReducingLoss
from functools import partial


def second_moment(weight, data):
    r"""
    Compute the second moment of a dataset.

    The second moment is computed as

    :math:`\frac{1}{\mathbf{1}^\intercal A \mathbf{1}} \mathbf{1}^\intercal A \circ \left (T - \frac{T \circ A}{A\mathbf{1}} \right )^2 \mathbf{1}`

    :Dimension: **weight :** :math:`(*, R, V)`
                    ``*`` denotes any number of preceding dimensions, R
                    denotes number of weights (e.g., regions of an atlas),
                    and V denotes number of locations (e.g., voxels).
                **data :** :math:`(*, V, T)`
                    T denotes number of observations at each location (e.g.,
                    number of time points).
    """
    mu = (weight @ data / weight.sum(-1, keepdim=True))
    diff = data.unsqueeze(-3) - mu.unsqueeze(-2)
    sigma = (diff * weight.unsqueeze(-1)) ** 2 / weight.sum()
    return sigma


class SecondMoment(ReducingLoss):
    def __init__(self, nu=1, reduction=None, name=None):
        reduction = reduction or mean
        super(SecondMoment, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=second_moment,
            name=name
        )
