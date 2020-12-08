# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Graph measures
~~~~~~~~~~~~~~
Measures on graphs and networks.
"""
import torch


def girvan_newman_null(A):
    k_i = A.sum(-1, keepdim=True)
    k_o = A.sum(-2, keepdim=True)
    two_m = k_i.sum(-2, keepdim=True)
    return k_i @ k_o / two_m


def modularity_matrix(A, gamma=1, null=girvan_newman_null,
                      normalise=False, **params):
    mod = A - gamma * null(A, **params)
    if normalise:
        two_m = A.sum([-2, -1], keepdim=True)
        return mod / two_m
    return mod
