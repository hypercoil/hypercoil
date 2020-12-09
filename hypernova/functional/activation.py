# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Activation functions
~~~~~~~~~~~~~~~~~~~~
Additional activation functions for neural network layers.
"""
import torch


def laplace(input, loc=0, width=1):
    """
    Double exponential activation function.

    The double exponential activation function is applied elementwise as

    :math:`e^{\frac{|x - \mu|}{b}}`

    to inputs x with centre :math:`\mu` and width b. It constrains its outputs
    to the range (0, 1], mapping values closer to its centre to larger outputs.
    It is Lipschitz continuous over the reals and differentiable except at its
    centre.

    Dimension
    ---------
    As this activation function is applied elementwise, it conserves dimension;
    the output will be of the same shape as the input.

    Parameters
    ----------
    input : Tensor
        Tensor to be transformed elementwise by the double exponential
        activation function.
    loc : float or broadcastable Tensor (default 0)
        Centre parameter :math:`\mu` of the double exponential function.
    width : float or broadcastable Tensor (default 1)
        Spread parameter b of the double exponential function.

    Returns
    -------
    out : Tensor
        Transformed input tensor.
    """
    return torch.exp(torch.abs(input - loc) / width)
