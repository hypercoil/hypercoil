# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Polynomial convolution
~~~~~~~~~~~~~~~~~~~~~~
Modules supporting polynomial convolution of time series and other data.
"""
import math
import torch
from torch.nn import Module, Parameter, init
from ..functional import polyconv2d
from ..init.laplace import LaplaceInit


class PolyConv2D(Module):
    """
    2D convolution over a polynomial expansion of an input signal.

    In a degree-K polynomial convolution, each channel of the input dataset is
    mapped across K channels, and raised to the ith power at the ith channel.
    The convolution kernel's ith input channel thus views the input dataset
    raised to the ith power.

    Dimension
    ---------
    - Input: :math:`(N, *, P, O)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      P denotes number of variables, O denotes number of observations.
    - Output: :math:`(N, *, C_{out}, P, O)`
      :math:`C_{out}` denotes number of output channels.

    Parameters
    ----------
    degree : int (default 2)
        Maximum degree of the polynomial expansion.
    out_channels : int (default 1)
        Number of output channels produced by the convolution.
    memory : int (default 3)
        Kernel memory. The number of previous observations viewable by the
        kernel.
    kernel_width : int (default 1)
        Number of adjoining variables simultaneously viewed by the kernel.
        Unless the variables are ordered and evenly sampled, this should either
        be 1 or P. Setting this equal to 1 applies the same kernel to all
        variables, while setting it equal to P applies a unique kernel for
        each variable.
    padding : int or None (default None)
        Number of zero-padding frames added to both sides of the input.
    bias : bool (default False)
        Indicates that a learnable bias should be added channel-wise to the
        output.
    include_const : bool (default False)
        Indicates that a constant term should be included in the polynomial
        expansion. This is almost equivalent to `bias`, and it is advised to
        use `bias` instead because it both is more efficient and exhibits more
        appropriate edge behaviours.
    future_sight : bool (default False)
        Indicates that the kernel should also view a number of observations
        equal to `memory` in the future.
    init : in-place callable (default LaplaceInit)
        Function for initialising the filter weight. By default, the filter
        weight is initialised as a discretised double exponential centred on
        the present time point and the first power such that it approximates
        identity, with a small amount of Gaussian noise added.

    Attributes
    ----------
    weight : Tensor :math:`(C_{out}, C_{in}, 2M + 1, W)`
        Learnable kernel weights for polynomial convolution. :math:`C_{out}` is
        the total number of learnable kernels; :math:`C_{in}` is the number of
        input channels (most often the maximum polynomial degree); M is the
        kernel memory, and W is the kernel width. The values are initialised
        following the `init_` function.
    bias : Tensor :math:`(C_{out})`
        Learnable bias for polynomial convolution. If `bias` is True, then the
        values of these weights are sampled from a uniform distribution whose
        bounds are the positive and negative inverse square root of the
        fan-out.
    mask : Tensor :math:`(C_{out}, C_{in}, 2M + 1, W)`
        Used to limit future sight by zeroing entries in the weight that
        correspond to future observations.
    """
    def __init__(self, degree=2, out_channels=1, memory=3, kernel_width=1,
                 padding=None, bias=False, include_const=False,
                 future_sight=False, init=None):
        super(PolyConv2D, self).__init__()

        self.in_channels = degree + include_const
        self.out_channels = out_channels
        self.memory = memory
        self.kernel_length = 2 * memory + 1
        self.kernel_width = kernel_width
        self.padding = padding
        self.degree = degree
        self.include_const = include_const
        self.future_sight = future_sight
        if include_const:
            self.init = init or LaplaceInit(loc=(1, 0, memory))
        else:
            self.init = init or LaplaceInit(loc=(0, 0, memory))

        self.weight = Parameter(torch.Tensor(
            self.out_channels,
            self.in_channels,
            self.kernel_width,
            self.kernel_length))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.init(self.weight)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

        if self.future_sight:
            self.mask = None
        else:
            self.mask = Parameter(torch.ones_like(self.weight))
            self.mask.requires_grad = False
            self.weight.requires_grad = False
            self.mask[:, :, :, (self.memory + 1):] = 0
            self.weight[:] = self.weight * self.mask
            self.weight.requires_grad = True

    def __repr__(self):
        s = (
            f'{self.__class__.__name__}(degree={self.degree}, out_channels='
            f'{self.out_channels}, memory={self.memory}'
        )
        if self.future_sight:
            s += f', future_sight={self.memory}'
        if self.kernel_width > 1:
            s += f', width={self.kernel_width}'
        if self.bias is not None:
            s += ', bias=True'
        s += ')'
        return s

    def forward(self, input):
        if self.future_sight:
            weight = self.weight
        else:
            weight = self.weight * self.mask
        return polyconv2d(
            input, weight=weight, bias=self.bias, padding=self.padding,
            include_const=self.include_const)
