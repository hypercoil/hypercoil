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


def _second_moment(weight, data, mu):
    """
    Core computation for second-moment loss.
    """
    diff = data.unsqueeze(-3) - mu.unsqueeze(-2)
    sigma = (diff * weight.unsqueeze(-1)) ** 2 / weight.sum()
    return sigma


def second_moment(weight, data, standardise=False):
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
    if standardise:
        data = (
            data - data.mean(-1, keepdim=True)) / data.std(-1, keepdim=True)
    mu = (weight @ data / weight.sum(-1, keepdim=True))
    return _second_moment(weight, data, mu)


def second_moment_centred(weight, data, mu,
                          standardise_data=False,
                          standardise_mu=False):
    r"""
    Compute the second moment of a dataset about a specified mean.
    """
    if standardise_data:
        data = (
            data - data.mean(-1, keepdim=True)) / data.std(-1, keepdim=True)
    if standardise_mu:
        mu = (mu - mu.mean(-1)) / mu.std(-1)
    return _second_moment(weight, data, mu)


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
    def __init__(self, nu=1, standardise=False, reduction=None, name=None):
        reduction = reduction or mean
        loss = partial(second_moment, standardise=standardise)
        super(SecondMoment, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=loss,
            name=name
        )


class SecondMomentCentred(ReducingLoss):
    r"""
    Compute the second moment of a dataset/linear mapping pair about a
    specified mean.
    """
    def __init__(self, nu=1, standardise_data=False, standardise_mu=False,
                 reduction=None, name=None):
        reduction = reduction or mean
        loss = partial(second_moment_centred,
                       standardise_data=standardise_data,
                       standardise_mu=standardise_mu)
        super(SecondMomentCentred, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=loss,
            name=name
        )
