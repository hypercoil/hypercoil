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
    r"""
    Compute the second moment of a dataset/linear mapping pair.

    The second moment is defined as

    :math:`\frac{1}{\mathbf{1}^\intercal A \mathbf{1}} \mathbf{1}^\intercal A \circ \left (T - \frac{T \circ A}{A\mathbf{1}} \right )^2 \mathbf{1}`

    Given an input dataset :math:`T` and a linear mapping :math:`A`, the
    second moment loss measures the extent to which :math:`A` tends to map
    similar inputs to the same output.

    Penalising this quantity can thus promote learning a linear mapping whose
    rows load onto similar feature sets.

    Parameters
    ----------
    nu : float (default 1)
        Loss function weight multiplier.
    reduction : callable (default `torch.mean`)
        Map from a tensor of arbitrary dimension to a scalar. The output of
        the second moment loss is passed into `reduction` to return a scalar.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.

    Notes
    -----
        This loss can have a very large memory footprint, because it requires
        computing an intermediate tensor with dimensions equal to the number
        of rows in the linear mapping, multiplied by the number of columns in
        the linear mapping, multiplied by the number of columns in the
        dataset.
    """
    def __init__(self, nu=1, reduction=None, name=None):
        reduction = reduction or mean
        super(SecondMoment, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=second_moment,
            name=name
        )
