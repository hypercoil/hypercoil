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


def multivariate_kurtosis(ts, l2=0, dimensional_scaling=False):
    if dimensional_scaling:
        d = ts.shape[-2]
        denom = d * (d + 2)
    else:
        denom = 1
    prec = precision(ts, l2=l2).unsqueeze(-3)
    ts = ts.transpose(-1, -2).unsqueeze(-2)
    maha = (ts @ prec @ ts.transpose(-1, -2)).squeeze()
    return -(maha ** 2).mean(-1) / denom


class MultivariateKurtosis(ReducingLoss):
    """
    Multivariate kurtosis following Mardia, as used by Laumann and colleagues.

    Parameters
    ----------
    nu : float (default 1)
        Loss function weight multiplier.
    l2 : float (default 0)
        L2 regularisation multiplier when computing the precision matrix.
    dimensional_scaling : bool (default False)
        The expected multivariate kurtosis for a normally distributed,
        stationary process of infinite duration with d channels (or variables)
        is :math:`d (d + 2)`. Setting this to true normalises for the process
        dimension by dividing the obtained kurtosis by :math:`d (d + 2)`. This
        has no effect in determining the optimum.
    reduction : callable (default ``torch.mean``)
        Map from a tensor of arbitrary dimension to a scalar. The vector of
        computed multivariate kurtosis values is passed into ``reduction`` to
        return a scalar.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
    """
    def __init__(self, nu=1, l2=0, dimensional_scaling=False,
                 reduction=None, name=None):
        reduction = reduction or torch.mean
        loss = partial(
            multivariate_kurtosis,
            l2=l2,
            dimensional_scaling=dimensional_scaling
        )
        super().__init__(
            nu=nu,
            reduction=reduction,
            loss=loss,
            name=name
        )
