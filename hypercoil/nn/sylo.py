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
from functools import partial
from ..functional import sylo, crosshair_similarity, delete_diagonal
from ..init.sylo import sylo_init_
from ..init.mpbl import BipartiteLatticeInit
from ..nn import Recombinator


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
    delete_diagonal: bool
        Delete the diagonal of the output.

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
                 symmetry=True, similarity=crosshair_similarity,
                 delete_diagonal=False, init=None, device=None, dtype=None):
        super(Sylo, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

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
        self.delete_diagonal = delete_diagonal
        self.init = init

        self.weight_L = Parameter(torch.empty(
            out_channels, in_channels, H, rank, **factory_kwargs
        ))
        if symmetry is True:
            self.weight_R = self.weight_L
        else:
            self.weight_R = Parameter(torch.empty(
                out_channels, in_channels, W, rank, **factory_kwargs
            ))

        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
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
        out = sylo(input, self.weight_L, self.weight_R,
                   self.bias, self.symmetry, self.similarity)
        if self.delete_diagonal:
            return delete_diagonal(out)
        return out


class SyloResBlock(nn.Module):
    """
    Sylo-based residual block by convolutional analogy. Patterned after
    torchvision's ``BasicBlock``. Restructured to follow principles from
    He et al. 2016, 'Identity Mappings in Deep Residual Networks'.
    Vertical compression module handles both stride-over-vertices and
    downsampling. It precedes all other operations and the specified dimension
    should be the compressed dimension.
    """
    def __init__(
        self,
        dim,
        in_channels,
        channels,
        nlin=None,
        recombine=True,
        norm_layer=None,
        compression=None
    ):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        nlin = nlin or partial(nn.ReLU, inplace=True)
        self.compression = compression
        if compression is not None:
            compression_out = compression.out_channels * in_channels
            if compression_out != channels:
                self.recombine1 = Recombinator(
                    in_channels=compression_out,
                    out_channels=channels,
                    bias=False
                )
            else:
                self.recombine1 = None
        else:
            self.recombine1 = None
        self.norm1 = norm_layer(channels)
        self.nlin = nlin(inplace=True)
        self.sylo1 = Sylo(
            channels,
            channels,
            dim,
            rank=3,
            symmetry='cross'
        )
        if recombine:
            self.recombine2 = Recombinator(
                in_channels=channels,
                out_channels=channels
            )
        else:
            self.recombine2 = None
        self.norm2 = norm_layer(channels)
        self.sylo2 = Sylo(
            channels,
            channels,
            dim,
            rank=3,
            symmetry='cross'
        )
        if recombine:
            self.recombine3 = Recombinator(
                in_channels=channels,
                out_channels=channels
            )
        else:
            self.recombine2 = None

    def forward(self, X):
        if self.compression is not None:
            X = self.compression(X)
        if self.recombine1 is not None:
            X = self.recombine1(X)
        identity = X

        out = self.norm1(X)
        out = self.nlin(out)
        out = self.sylo1(out)
        if self.recombine2 is not None:
            out = self.recombine2(out)
        out = self.norm2(out)
        out = self.nlin(out)
        out = self.sylo2(out)
        if self.recombine3 is not None:
            out = self.recombine3(out)

        out = out + identity
        return out


class SyloBottleneck(nn.Module):
    def __init__(
        self,
        dim,
        in_channels,
        channels,
        nlin=None,
        recombine=True,
        norm_layer=None,
        compression=None
    ):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        nlin = nlin or partial(nn.ReLU, inplace=True)
        raise NotImplementedError('Bottleneck analogy is incomplete')


class SyloResNetScaffold(nn.Module):
    def __init__(
        self,
        in_dim,
        in_channels,
        dim_sequence=None,
        channel_sequence=(16, 32, 64, 128),
        block_sequence=(33, 33, 33, 33),
        block=SyloResBlock,
        recombine=True,
        norm_layer=None,
        nlin=None,
        compressions=None
    ):
        super().__init__()
        norm_layer = norm_layer or nn.BatchNorm2d
        nlin = nlin or partial(nn.ReLU, inplace=True)
        dim_sequence = dim_sequence or (
            in_dim // 2,
            in_dim // 4,
            in_dim // 8,
            in_dim // 16)
        in_channel_sequence = (
            [channel_sequence[0]] + list(channel_sequence[:-1]))
        #if compressions is not None:
        #    multipliers = [v.out_channels for v in compressions]
        #    out_channel_sequence = [None for _ in multipliers]
        #    for i, (c, m) in enumerate(zip(in_channel_sequence, multipliers)):
        #        assert c / m == c // m, (
        #            f'Number of block channels {c} must be an even multiple '
        #            f'of compression channel amplification factor {m}')
        #        out_channel_sequence[i] = c // m
        compressions = compressions or [None for _ in dim_sequence]
        self._norm_layer = norm_layer
        self._nlin = nlin

        self.channel_sequence = channel_sequence
        # TODO: revisit after adding channel groups to sylo

        sylo_in = []
        if isinstance(in_channels, int):
            in_channels = [(in_channels, channel_sequence[0])]
        for i, (c_in, c_out) in enumerate(in_channels):
            sylo_in += [Sylo(
                c_in,
                c_out,
                in_dim,
                rank=1,
                bias=False,
                symmetry='cross'
            )]
        self.sylo1 = nn.ModuleList(sylo_in)
        self.norm1 = norm_layer(channel_sequence[0])
        self.nlin = nlin()
        layers = [self._make_layer(block, i, c, b, d, v)
                  for i, c, b, d, v in zip(in_channel_sequence,
                                           channel_sequence,
                                           block_sequence,
                                           dim_sequence,
                                           compressions)]
        self.layers = nn.ModuleList(layers)

    def _make_layer(self, block, in_channels,
                    channels, blocks, dim, compression):
        norm_layer = self._norm_layer
        nlin = self._nlin

        layers = []
        layers.append(block(
            dim=dim,
            in_channels=in_channels,
            channels=channels,
            norm_layer=norm_layer,
            compression=compression
        ))
        for _ in range(1, blocks):
            layers.append(block(
                dim=dim,
                in_channels=channels,
                channels=channels,
                norm_layer=norm_layer,
                compression=None
            ))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = [s(x) for s in self.sylo1]
        x = torch.cat(x, -3)
        x = self.norm1(x)
        x = self.nlin(x)
        for i, l in enumerate(self.layers):
            x = l(x)
        return x


class SyloResNet(nn.Module):
    def __init__(self,
                 in_channels,
                 in_dim,
                 dim_sequence,
                 channel_sequence,
                 block_sequence,
                 lattice_order_sequence,
                 n_lattices,
                 channel_multiplier=1,
                 compression_init='svd',
                 recombine=True,
                 norm_layer=None,
                 nlin=None,
                 block=SyloResBlock,
                 potentials=None):
        super().__init__()
        init_arg = [{}]
        if init == 'svd':
            init_arg = [{'svd' : True}]
        elif init == 'resid':
            init_arg = [{'residualise' : True}]
        init_arg += [{} for _ in order_sequence[1:]]

        self._compression_specs = [
            BipartiteLatticeInit(channel_multiplier=channel_multiplier,
                                 n_lattices=n_lattices,
                                 n_out=n_out,
                                 order=order,
                                 **init)
            for n_out, order, init in zip(
                dim_sequence,
                lattice_order_sequence,
                init_arg
        )]

        for i, j in zip(self._compression_specs[:-1],
                        self._compression_specs[1:]):
            i.next = j
        if potentials is not None:
            self.set_potentials(potentials)

        all_dim_sequence = (
            [in_dim] + list(dim_sequence))
        self._compressions = [
            VerticalCompression(
                init=spec,
                in_features=_in,
                out_features=_out
            )
            for spec, (_in, _out) in
            zip(self._compression_specs,
                zip(all_dim_sequence[:-1],
                    all_dim_sequence[1:]))
        ]

        self.model = SyloResNetScaffold(
            in_dim=in_dim,
            in_channels=in_channels,
            dim_sequence=dim_sequence,
            channel_sequence=channel_sequence,
            block_sequence=block_sequence,
            block=block,
            recombine=recombine,
            norm_layer=norm_layer,
            nlin=nlin,
            compressions=self._compressions
        )

    def set_potentials(self, potentials):
        self._compression_specs[0].set_potentials(cor)
        try:
            for c in self._compressions:
                c.reset_parameters()
        except AttributeError:
            pass

    def forward(self, x):
        return self.model(x)
