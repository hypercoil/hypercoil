# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility
~~~~~~~
There are some extremely simple and basic things that it seems torch is
absolutely terrible at handling. For these situations, we have this hideous,
disorganised group of utility functions. Hopefully someday they can disappear
altogether out of irrelevance, but for now they exist, a sad blemish.
"""
import torch


def conform_mask(tensor, msk, axis, batch=False):
    """
    Conform a mask or weight for elementwise applying to a tensor.
    """
    if batch:
        tile = list(tensor.shape)
        tile[0] = 1
        tile[axis] = 1
        shape = [1 for _ in range(tensor.dim())]
        shape[0] = msk.shape[0]
        shape[axis] = msk.shape[-1]
        msk = msk.view(*shape).tile(*tile)
    else:
        shape_pfx = tensor.shape[:axis]
        msk = msk.tile(*shape_pfx, 1)
    return msk



def mask(tensor, msk, axis):
    """
    Mask a tensor along an axis.
    """
    shape_pfx = tensor.shape[:axis]
    if axis == -1:
        shape_sfx = ()
    else:
        shape_sfx = tensor.shape[(axis + 1):]
    msk = msk.tile(*shape_pfx, 1)
    return tensor[msk].reshape(*shape_pfx, -1, *shape_sfx)


def wmean(input, weight, dim=None, keepdim=False):
    """
    Reducing function for reducing losses: weighted mean.
    """
    if dim is None:
        dim = list(range(input.dim()))
    elif isinstance(dim, int):
        dim = (dim,)
    assert weight.dim() == len(dim), (
        'Weight must have as many dimensions as are being reduced')
    retain = [True for _ in range(input.dim())]
    for d in dim:
        retain[d] = False
    for i, d in enumerate(retain):
        if d: weight = weight.unsqueeze(i)
    wtd = (weight * input)
    return wtd.sum(dim, keepdim=keepdim) / weight.sum(dim, keepdim=keepdim)
