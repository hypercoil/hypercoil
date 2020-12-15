# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Laplace initialisation
~~~~~~~~~~~~~~~~~~~~~~
Initialise parameters to match a double exponential function.
"""
import torch
from functools import reduce


def laplace_init_(tensor, loc=None, width=None, norm=None,
                  var=0.02, excl_axis=None):
    """
    Laplace initialisation.

    Initialise a tensor such that its values are interpolated by a
    multidimensional double exponential function, :math:`e^{-|x|}`.

    Parameters
    ----------
    tensor : Tensor
        Tensor to initialise in-place.
    loc : iterable or None (default None)
        Origin point of the double exponential, in array coordinates. If None,
        this will be set to the centre of the array.
    width : iterable or None (default None)
        Decay rate of the double exponential along each array axis. If None,
        this will be set to 1 isotropically. If this is very large, the result
        will approximate a delta function at the specified `loc`.
    norm : 'max', 'sum', or None (default None)
        Normalisation to apply to the output.
        - 'max' divides the output by its maximum value such that the largest
          value in the initialised tensor is exactly 1.
        - 'sum' divides the output by its sum such that all entries in the
          initialised tensor sum to 1.
        - None indicates that the output should not be normalised.
    var : float
        Variance of the Gaussian distribution from which the random noise is
        sampled.
    excl_axis : list or None (default None)
        List of axes across which a double exponential is not computed. Instead
        the double exponential computed across the remaining axes is broadcast
        across the excluded axes.

    Returns
    -------
    None. The input tensor is initialised in-place.
    """
    rg = tensor.requires_grad
    tensor.requires_grad = False
    loc = loc or [(x - 1) / 2 for x in tensor.size()]
    width = width or [1 for _ in range(tensor.dim())]
    width = torch.Tensor(width)
    dim = len(loc)
    excl_axis = excl_axis or []
    axes = []
    for ax, l, w in zip(tensor.size()[-dim:], loc, width[-dim:]):
        new_ax = torch.arange(-l, -l + ax)
        new_ax = torch.exp(-torch.abs(new_ax) / w)
        axes += [new_ax]
    shape = [-1]
    val = []
    for i, ax in enumerate(reversed(axes)):
        if -(i + 1) not in excl_axis and (dim - i - 1) not in excl_axis:
            val = [ax.view(shape)] + val
        shape += [1]
    val = reduce(torch.multiply, val)
    if norm == 'max':
        val /= val.max()
    elif norm == 'sum':
        val /= val.sum()
    val.type(tensor.dtype)
    if var != 0:
        val = val + torch.randn(tensor.size()) * var
    tensor.copy_(val)
    tensor.requires_grad = rg
