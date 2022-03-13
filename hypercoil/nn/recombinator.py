# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Linear recombinator layer for feature-map learning networks.
A 1x1 conv layer by another name.
"""
import math
import torch
from torch import nn
from torch.nn import init, Parameter


class Recombinator(nn.Module):
    """Linear recombinator layer for feature maps. It should also be possible
    to substitute a 1x1 convolutional layer with similar results.

    Parameters
    ----------
    in_channels: int
        Number of channels or feature maps input to the recombinator layer.
    out_channels: int
        Number of recombined channels or feature maps output by the
        recombinator layer.
    bias: bool
        If True, adds a learnable bias to the output.
    positive_only: bool (default False)
        If True, initialise with only positive weights.
    init: dict
        Dictionary of parameters to pass to the Kaiming initialisation
        function.
        Default: {'nonlinearity': 'linear'}

    Attributes
    ----------
    weight: Tensor
        The learnable mixture matrix of the module of shape
        `in_channels` x `out_channels`.
    bias: Tensor
        The learnable bias of the module of shape `out_channels`.
    """
    __constants__ = ['in_channels', 'out_channels', 'weight', 'bias']

    def __init__(self, in_channels, out_channels,
                 bias=True, positive_only=False, init=None):
        super(Recombinator, self).__init__()

        if init is None:
            init = {'nonlinearity': 'linear'}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.positive_only = positive_only
        self.init = init

        self.weight = Parameter(torch.Tensor(out_channels, in_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, **self.init)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        if self.positive_only:
            rg = self.weight.requires_grad
            self.weight.requires_grad = False
            self.weight[:] = torch.abs(self.weight)
            self.weight.requires_grad = rg

    def extra_repr(self):
        s = 'in_channels={}, out_channels={}'.format(
            self.in_channels, self.out_channels)
        if self.bias is None:
            s += ', bias=False'
        return s

    def forward(self, input):
        return recombine(input, self.weight, self.bias)


# TODO: Move to functional if we decide to keep this instead of just using 1x1
def recombine(input, mixture, bias=None):
    """Create a new mixture of the input feature maps.

    Parameters
    ----------
    input: Tensor (N x C_in x H x W)
        Stack of input matrices or feature maps.
    mixture: Tensor (C_out x C_in)
        Mixture matrix or recombinator.
    bias: Tensor (C_in)
        Bias term to apply after recombining.
    """
    output = (mixture @ input.transpose(1, 2)).transpose(1, 2)
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    return output