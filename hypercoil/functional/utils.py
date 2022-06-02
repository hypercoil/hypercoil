# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utility
~~~~~~~
A hideous, disorganised group of utility functions. Hopefully someday they
can disappear altogether or be moved elsewhere, but for now they exist, a sad
blemish.
"""
import torch


def conform_mask(tensor, msk, axis, batch=False):
    """
    Conform a mask or weight for elementwise applying to a tensor.

    There is almost certainly a better way to do this.
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


def complex_decompose(complex):
    """
    Decompose a complex-valued tensor into amplitude and phase components.

    Dimension
    ---------
    Each output is of the same shape as the input.

    Parameters
    ----------
    complex : Tensor
        Complex-valued tensor.

    Returns
    -------
    ampl : Tensor
        Amplitude of each entry in the input tensor.
    phase : Tensor
        Phase of each entry in the input tensor, in radians.
    """
    ampl = torch.abs(complex)
    phase = torch.angle(complex)
    return ampl, phase


def complex_recompose(ampl, phase):
    """
    Reconstitute a complex-valed tensor from real-valued tensors denoting its
    amplitude and its phase.

    Dimension
    ---------
    Both inputs must be the same shape (or broadcastable). The output is the
    same shape as the inputs.

    Parameters
    ----------
    ampl : Tensor
        Real-valued array storing complex number amplitudes.
    phase : Tensor
        Real-valued array storing complex number phases in radians.

    Returns
    -------
    complex : Tensor
        Complex numbers formed from the specified amplitudes and phases.
    """
    # TODO : consider using the complex exponential when torch enables it,
    # depending on the gradient properties
    # see here : https://discuss.pytorch.org/t/complex-functions-exp-does- ...
    # not-support-automatic-differentiation-for-outputs-with-complex-dtype/98039
    # Supposedly it was updated, but it still isn't working after calling
    # pip install torch --upgrade
    # (old note, might be working now)
    # https://github.com/pytorch/pytorch/issues/43349
    # https://github.com/pytorch/pytorch/pull/47194
    return ampl * (torch.cos(phase) + 1j * torch.sin(phase))
    #return ampl * torch.exp(phase * 1j)
