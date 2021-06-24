# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Centre of mass
~~~~~~~~~~~~~~
Differentiably compute a weight's centre of mass.
"""
import torch


def cmass(X, axes=None, na_rm=False):
    dim = X.size()
    ndim = X.dim()
    axes = axes or range(ndim)
    out_dim = [s for ax, s in enumerate(dim) if ax not in axes]
    out_dim += [len(axes)]
    out = torch.zeros(out_dim)
    for i, ax in enumerate(axes):
        coor = torch.arange(1, X.size(ax) + 1)
        while coor.dim() < ndim - ax:
            coor.unsqueeze_(-1)
        num = (coor * X).sum(axes)
        denom = X.sum(axes)
        out[..., i] = num / denom
        if na_rm is not False:
            out[denom==0, i] = na_rm
    return out
