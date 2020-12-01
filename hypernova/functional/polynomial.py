# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Polynomial convolution
~~~~~~~~~~~~~~~~~~~~~~
Functions supporting polynomial convolution of time series and other data.
"""
import torch


def polychan(X, degree=2, include_const=False):
    """
    Create a polynomial channel basis for the data.

    Single-channel data are mapped across K channels, and raised to the ith
    power at the ith channel.

    Dimension
    ---------
    - Input: :math:`(N, *, C, obs)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      C denotes number of data channels or variables, obs denotes number of
      observations
    - Output: :math:`(N, K, *, C, obs)`
      K denotes maximum polynomial degree to include

    Parameters
    ----------
    X : Tensor
        Dataset to expand as a polynomial basis. A new channel will be created
        containing the same dataset raised to each power up to the specified
        degree.
    degree : int >= 2 (default 2)
        Maximum polynomial degree to be included in the output basis.
    include_const : bool (default False)
        Indicates that a constant or intercept term corresponding to the zeroth
        power should be included.

    Returns
    -------
    out : Tensor
        Input dataset expanded as a K-channel polynomial basis.
    """
    if X.dim() > 2:
        pass
    elif X.dim() == 2:
        X = X.view(1, *X.size())
    elif X.dim() == 1:
        X = X.view(1, -1)
    stack = [X]
    for _ in range(degree - 1):
        stack += [stack[-1] * X]
    if include_const:
        stack = [torch.ones_like(X)] + stack
    return torch.stack(stack, 1)


def polyconv(X, weight, include_const=False, bias=None):
    degree = weight.size(1) - include_const
    padding = (0, weight.size(-1) // 2)
    X = polychan(X, degree=degree, include_const=include_const)
    return torch.conv2d(X, weight, bias=bias, stride=1, padding=padding)
