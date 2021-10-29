# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Sylo
~~~~
Sylo ("symmetric low-rank") kernel operator.
"""
import math
import torch
from torch import nn
from torch.nn import init, Parameter
from ..functional import sylo, crosshair_similarity
from ..init.sylo import sylo_init_


class Sylo(nn.Module):
    """
    Layer that learns a set of (possibly) symmetric, low-rank representations
    of a dataset.

    Parameters
    ----------
    in_channels: int
        Number of channels or layers in the input graph or matrix.
    out_channels: int
        Number of channels or layers in the output graph or matrix. This is
        equal to the number of learnable templates.
    dim: int or tuple(int)
        Number of vertices in the graph or number of columns in the matrix.
        If the graph is bipartite or the matrix is nonsquare, then this should
        be a 2-tuple.
    rank: int
        Rank of the templates learned by the sylo module. Default: 1.
    bias: bool
        If True, adds a learnable bias to the output. Default: True
    symmetry: bool, 'cross', or 'skew'
        Symmetry constraints to impose on learnable templates.
        * If False, no symmetry constraints are placed on the templates learned
          by the module.
        * If True, the module is constrained to learn symmetric representations
          of the graph or matrix: the left and right generators of each
          template are constrained to be identical.
        * If 'cross', the module is also constrained to learn symmetric
          representations of the graph or matrix. However, in this case, the
          left and right generators can be different, and the template is
          defined as the average of the expansion and its transpose:
          1/2 (L @ R.T + R @ L.T).
        * If 'skew', the module is constrained to learn skew-symmetric
          representations of the graph or matrix. The template is defined as
          the difference between the expansion and its transpose:
          L @ R.T - R @ L.T
        This option is not available for nonsquare matrices or bipartite
        graphs. Note that the parameter count doubles if this is False.
        Default: True
    similarity: function
        Definition of the similarity metric. This must be a function whose
        inputs and outputs are:
        * input 0: reference matrix (N x C_in x H x W)
        * input 1: left template generator (C_out x C_in x H x R)
        * input 2: right template generator (C_out x C_in x H x R)
        * input 3: symmetry constraint ('cross', 'skew', or other)
        * output (N x C_out x H x W)
        Similarity is computed between each of the N matrices in the first
        input stack and the (low-rank) matrix derived from the outer-product
        expansion of the second and third inputs.
        Default: `crosshair_similarity`
    init: dict
        Dictionary of parameters to pass to the sylo initialisation function.
        Default: {'nonlinearity': 'relu'}

    Attributes
    ----------
    weight: Tensor
        The learnable weights of the module of shape
        `out_channels` x `in_channels` x `dim` x `rank`.
    bias: Tensor
        The learnable bias of the module of shape `out_channels`.
    """
    __constants__ = ['in_channels', 'out_channels', 'H', 'W', 'rank', 'bias']

    def __init__(self, in_channels, out_channels, dim, rank=1, bias=True,
                 symmetry=True, similarity=crosshair_similarity, init=None):
        super(Sylo, self).__init__()

        if isinstance(dim, int):
            H, W = dim, dim
        elif symmetry and dim[0] != dim[1]:
            raise ValueError('Symmetry constraints are invalid for nonsquare '
                             'matrices. Set symmetry=False or use an integer '
                             'dim')
        else:
            H, W = dim
        if init is None:
            init = {'nonlinearity': 'relu'}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.dim = (H, W)
        self.symmetry = symmetry
        self.similarity = similarity
        self.init = init

        self.weight_L = Parameter(
            torch.Tensor(out_channels, in_channels, H, rank))
        if symmetry is True:
            self.weight_R = self.weight_L
        else:
            self.weight_R = Parameter(
                torch.Tensor(out_channels, in_channels, W, rank))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        sylo_init_((self.weight_L, self.weight_R),
                   symmetry=self.symmetry, **self.init)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_L)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def __repr__(self):
        s = '{}(dim={}, in_channels={}, out_channels={}, rank={}'.format(
            self.__class__.__name__, self.dim,
            self.in_channels, self.out_channels, self.rank)
        if self.bias is None:
            s += ', bias=False'
        if self.symmetry is True:
            s += ', symmetric'
        elif self.symmetry == 'cross':
            s += ', cross-symmetric'
        elif self.symmetry == 'skew':
            s += ', skew-symmetric'
        s += ')'
        return s

    def forward(self, input):
        return sylo(input, self.weight_L, self.weight_R,
                    self.bias, self.symmetry, self.similarity)

