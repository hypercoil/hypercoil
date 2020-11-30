# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Symmetric matrix powers
~~~~~~~~~~~~~~~~~~~~~~~
Differentiable computation of matrix logarithm, exponential, and square root.
For use with symmetric, positive semidefinite matrices.
"""
import torch


def symxfm(input, xfm, spd=True):
    """
    Apply a specified matrix-valued transformation to a batch of symmetric
    (probably positive semidefinite) tensors.

    Dimension
    ---------
    - Input: :math:`(N, *, D, D)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      D denotes matrix row and column dimension
    - Output: :math:`(N, *, D, D)`

    Parameters
    ----------
    input : Tensor
        Batch of symmetric tensors to transform.
    xfm : dimension-conserving torch callable
        Transformation to apply as a matrix-valued function
    spd : bool (default True)
        Indicates that the matrices in the input batch are symmetric positive
        semidefinite; guards against numerical rounding errors and ensures all
        eigenvalues are nonnegative.

    Returns
    -------
    output : Tensor
        Transformation of each matrix in the input batch.
    """
    L, Q = torch.symeig(input, eigenvectors=True)
    if spd:
        L = torch.maximum(L, torch.zeros_like(L))
    Lxfm = torch.diag_embed(xfm(L))
    return Q @ Lxfm @ Q.transpose(-1, -2)


def symlog(input):
    return symxfm(input, torch.log)


def symexp(input):
    return symxfm(input, torch.exp)


def symsqrt(input):
    return symxfm(input, torch.sqrt)
