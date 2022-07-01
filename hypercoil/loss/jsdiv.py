# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Loss functions using the Jensen-Shannon divergence.
"""
import torch
from torch.nn.functional import kl_div
from functools import partial
from .base import ReducingLoss


def js_divergence(P, Q, axis=-1, keepdim=True):
    M = 0.5 * (P + Q)
    M_log = M.log()
    div = (kl_div(M_log, P, reduction='none') +
           kl_div(M_log, Q, reduction='none')) / 2
    if axis is None:
        return div
    return div.sum(dim=axis, keepdim=keepdim)


def js_divergence_logit(P, Q, axis=-1, keepdim=True):
    prob_axis = axis
    if prob_axis is None:
        prob_axis = -1
    P = torch.softmax(P, prob_axis)
    Q = torch.softmax(Q, prob_axis)
    return js_divergence(P, Q, axis=axis, keepdim=keepdim)


class JSDivergence(ReducingLoss):
    """
    Jensen-Shannon divergence, applied to probabilities.

    Use :class:`SoftmaxJSDivergence` for the logit version.

    Parameters
    ----------
    nu : float (default 1)
        Loss function weight multiplier.
    reduction : callable (default ``torch.mean``)
        Map from a tensor of arbitrary dimension to a scalar. The vector of
        JS divergences computed for each distribution is passed into
        ``reduction`` to return a scalar.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
    """
    def __init__(self, nu=1, axis=-1, reduction=None, name=None):
        reduction = reduction or torch.mean
        super().__init__(
            nu=nu,
            reduction=reduction,
            loss=partial(js_divergence, axis=axis),
            name=name
        )


class SoftmaxJSDivergence(ReducingLoss):
    """
    Jensen-Shannon divergence, applied to logits.

    Use :class:`JSDivergence` for the probability version.

    Parameters
    ----------
    nu : float (default 1)
        Loss function weight multiplier.
    reduction : callable (default ``torch.mean``)
        Map from a tensor of arbitrary dimension to a scalar. The vector of
        JS divergences computed for each distribution is passed into
        ``reduction`` to return a scalar.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
    """
    def __init__(self, nu=1, axis=-1, reduction=None, name=None):
        reduction = reduction or torch.mean
        super().__init__(
            nu=nu,
            reduction=reduction,
            loss=partial(js_divergence_logit, axis=axis),
            name=name
        )
