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


def apply_mask(tensor, msk, axis):
    """
    Mask a tensor along an axis.
    """
    shape_pfx = tensor.shape[:axis]
    if axis == -1:
        shape_sfx = ()
    else:
        shape_sfx = tensor.shape[(axis + 1):]
    msk = msk.tile(*shape_pfx, 1)
    return tensor[msk].view(*shape_pfx, -1, *shape_sfx)


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


def selfwmean(input, dim=None, keepdim=False, gradpath='input', softmax=True):
    """
    Self-weighted mean reducing function. Completely untested. Will break and
    probably kill you in the process.
    """
    i = input.clone()
    w = input.clone()
    if softmax:
        w = torch.softmax(w)
    if gradpath == 'input':
        w = w.detach()
    elif gradpath == 'weight':
        i = i.detach()
    return wmean(input=i, weight=w, keepdim=keepdim, gradpath=gradpath)


# torch is actually very, very good at doing this. Looks like we might have
# miscellaneous utilities.
# It's not even continuous, let alone differentiable. Let's not use this.
def threshold(input, threshold, dead=0, leak=0):
    if not isinstance(dead, torch.Tensor):
        dead = torch.tensor(dead, dtype=input.dtype, device=input.device)
    if leak == 0:
        return torch.where(input > threshold, input, dead)
    return torch.where(input > threshold, input, dead + leak * input)
