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


def js_divergence(P, Q):
    M = 0.5 * (P + Q)
    M_log = M.log()
    return (kl_div(M_log, P.log(), reduction='none', log_target=True) +
            kl_div(M_log, Q.log(), reduction='none', log_target=True)) / 2


def js_divergence_logit(P, Q):
    M = 0.5 * (torch.softmax(P, -1) + torch.softmax(Q, -1))
    M_log = M.log()
    return (kl_div(M_log, P, reduction='none', log_target=True) +
            kl_div(M_log, Q, reduction='none', log_target=True)) / 2


class JSDivergence(ReducingLoss):
    def __init__(self, nu=1, reduction=None, name=None):
        super(JSDivergence).__init__(
            nu=nu,
            reduction=reduction,
            loss=js_divergence,
            name=name
        )


class JSDivergenceSoftmax(ReducingLoss):
    def __init__(self, nu=1, reduction=None, name=None):
        super(JSDivergence).__init__(
            nu=nu,
            reduction=reduction,
            loss=js_divergence_logit,
            name=name
        )
