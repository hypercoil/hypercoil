# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
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
from .recombinator import Recombinator
from .vertcom import VerticalCompression


class Sylo(nn.Module):
    r"""
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
    symmetry: bool, ``'cross'``, or ``'skew'`` (default True)
        Symmetry constraints to impose on learnable templates.

        * If False, no symmetry constraints are placed on the templates
          learned by the module.
        * If True, the module is constrained to learn symmetric
          representations of the graph or matrix: the left and right
          generators of each template are constrained to be identical.
        * If ``'cross'``, the module is also constrained to learn symmetric
          representations of the graph or matrix. However, in this case, the
          left and right generators can be different, and the template is
          defined as the average of the expansion and its transpose:
          :math:`\frac{1}{2} \left(L R^{\intercal} + R L^{\intercal}\right)`
        * If ``'skew'``, the module is constrained to learn skew-symmetric
          representations of the graph or matrix. The template is defined as
          the difference between the expansion and its transpose:
          :math:`\frac{1}{2} \left(L R^{\intercal} - R L^{\intercal}\right)`

        This option is not available for nonsquare matrices or bipartite
        graphs. Note that the parameter count doubles if this is False.
    coupling: None, ``'+'``, ``'-'``, ``'split'``, ``'learnable'``, ``'learnable_all'``
        Coupling parameter when expanding outer-product template banks.

        * A value of ``None`` disables the coupling parameter.
        * ``'+'`` is equivalent to ``None``, fixing coupling to positive 1.
          For ``symmetry=True``, this enforces positive semidefinite
          templates.
        * ``'-'`` fixes coupling parameters to negative 1. For
          ``symmetry=True``, this enforces negative semidefinite templates.
        * ``'split'`` splits channels such that approximately an equal number
          have coupling parameters fixed to +1 and -1. For ``symmetry=True``,
          this splits channels among positive and negative semidefinite
          templates. This option can also be useful when imposing a unilateral
          normed penalty to favour nonnegative weights, as the template bank
          can simultaneously satisfy the soft nonnegativity constraint and
          respond with positive activations to features of either sign,
          enabling these activations to survive downstream rectifiers.
        * A float value in (0, 1) is just like ``split`` but fixes the
          fraction of negative parameters to approximately the specified
          value.
        * Similarly, an int value fixes the number of negative output channels
          to the specified value.
        * ``'learnable'`` sets the diagonal terms of the coupling parameter
          (the coupling between vector 0 of the left generator and vector 0
          of the right generator, for instance, but not between vector 0 of
          the left generator and vector 1 of the right generator) to be
          learnable.
        * ``'learnable_all'`` sets all terms of the coupling parameter to be
        learnable.
    similarity: function
        Definition of the similarity metric. This must be a function whose
        inputs and outputs are:

        * input 0: reference matrix ``(N x C_in x H x W)``
        * input 1: left template generator ``(C_out x C_in x H x R)``
        * input 2: right template generator ``(C_out x C_in x H x R)``
        * input 3: symmetry constraint (``'cross'``, ``'skew'``, or other)
        * output ``(N x C_out x H x W)``

        Similarity is computed between each of the N matrices in the first
        input stack and the (low-rank) matrix derived from the outer-product
        expansion of the second and third inputs.
        Default: ``crosshair_similarity``
    init: dict
        Dictionary of parameters to pass to the sylo initialisation function.
        Default: ``{'nonlinearity': 'relu'}``
    delete_diagonal: bool
        Delete the diagonal of the output.

    Attributes
    ----------
    weight: Tensor
        The learnable weights of the module of shape
        ``out_channels x in_channels x dim x rank``.
    bias: Tensor
        The learnable bias of the module of shape ``out_channels``.
    """
    __constants__ = ['in_channels', 'out_channels', 'H', 'W', 'rank', 'bias']

    def __init__(self, in_channels, out_channels, dim, rank=1, bias=True,
                 symmetry=True, coupling=None,
                 similarity=crosshair_similarity,
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
            (out_channels, in_channels, H, rank), **factory_kwargs
        ))
        if symmetry is True:
            self.weight_R = self.weight_L
        else:
            self.weight_R = Parameter(torch.empty(
                (out_channels, in_channels, W, rank), **factory_kwargs
            ))

        if bias:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self._cfg_coupling(coupling, factory_kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        sylo_init_((self.weight_L, self.weight_R),
                   symmetry=self.symmetry, **self.init)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_L)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _cfg_coupling(self, coupling, factory_kwargs):
        if coupling == 'split':
            coupling = self.out_channels // 2
        elif isinstance(coupling, float):
            coupling = int(self.out_channels * coupling)
        if coupling is None or coupling == '+':
            self.coupling = None
        elif coupling == '-':
            self.coupling = Parameter(-torch.ones(
                (self.out_channels, self.in_channels, self.rank, 1),
                **factory_kwargs
            ), requires_grad=False)
        elif isinstance(coupling, int):
            self.coupling = Parameter(torch.ones(
                (self.out_channels, self.in_channels, self.rank, 1),
                **factory_kwargs
            ), requires_grad=False)
            self.coupling[:coupling] *= -1
        elif coupling == 'learnable':
            self.coupling = Parameter(torch.randn(
                (self.out_channels, self.in_channels, self.rank, 1),
                **factory_kwargs
            ))
        elif coupling == 'learnable_all':
            self.coupling = Parameter(torch.randn(
                (self.out_channels, self.in_channels, self.rank, self.rank),
                **factory_kwargs
            ))

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
        out = sylo(
            X=input,
            L=self.weight_L,
            R=self.weight_R,
            C=self.coupling,
            bias=self.bias,
            symmetry=self.symmetry,
            similarity=self.similarity)
        if self.delete_diagonal:
            return delete_diagonal(out)
        return out


class SyloNetworkScaffold(nn.Module):
    def __init__(
        self,
        dim_sequence,
        channel_sequence,
        recombine=True,
        norm_layer=None,
        nlin=None,
        compressions=None,
    ):
        super().__init__()
        nlin = nlin or partial(nn.ReLU, inplace=True)
        norm_layer = norm_layer or torch.nn.Identity
        compressions = compressions or [None]
        channels_io = zip(channel_sequence[:-1], channel_sequence[1:])
        dim_io = zip(dim_sequence[:-1], dim_sequence[1:])
        layers = []
        c_idx = 0
        for (in_channels, out_channels), (in_dim, out_dim) in zip(
            channels_io, dim_io):
            layers += [Sylo(
                in_channels=in_channels,
                out_channels=out_channels,
                dim=in_dim,
                rank=1,
                bias=True,
                symmetry='cross',
                coupling='split'
            )]
            if recombine:
                layers += [Recombinator(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    bias=False
                )]
            layers += [norm_layer(out_channels)]
            if in_dim != out_dim:
                layers += [compressions[c_idx]]
                c_idx += 1
        self._norm_layer = type(norm_layer(0))
        self.layers = nn.ModuleList(layers)
        self.nlin = nlin()

    def forward(self, input, query=None):
        query_idx = 0
        x = input
        for l in self.layers:
            if isinstance(l, Recombinator) and query is not None:
                x = l(x, query=query[query_idx])
                query_idx += 1
            else:
                x = l(x)
            if isinstance(l, self._norm_layer):
                x = self.nlin(x)
        return x


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
        recombine=True,
        nlin=None,
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
        self.nlin = nlin()
        self.sylo1 = Sylo(
            channels,
            channels,
            dim,
            rank=3,
            symmetry='cross',
            coupling='split'
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
            symmetry='cross',
            coupling='split'
        )
        if recombine:
            self.recombine3 = Recombinator(
                in_channels=channels,
                out_channels=channels
            )
        else:
            self.recombine2 = None

    def forward(self, X, query=None):
        if self.compression is not None:
            X = self.compression(X)
        if self.recombine1 is not None:
            X = self.recombine1(X)
        identity = X

        out = self.norm1(X)
        out = self.nlin(out)
        out = self.sylo1(out)
        if self.recombine2 is not None:
            if query is None:
                out = self.recombine2(out)
            else:
                out = self.recombine2(out, query=query[0])
        out = self.norm2(out)
        out = self.nlin(out)
        out = self.sylo2(out)
        if self.recombine3 is not None:
            if query is None:
                out = self.recombine3(out)
            else:
                out = self.recombine3(out, query=query[1])

        out = out + identity
        return out


class SyloBottleneck(nn.Module):
    def __init__(
        self,
        dim,
        in_channels,
        channels,
        recombine=True,
        nlin=None,
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
        compressions=None,
        community_dim=0
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
        self._recombine = recombine

        self.channel_sequence = channel_sequence
        # TODO: revisit after adding channel groups to sylo

        out_channels = channel_sequence[0]
        sylo_in = {'main': Sylo(
            in_channels,
            (out_channels - community_dim),
            in_dim,
            rank=1,
            bias=False,
            symmetry='cross',
            coupling='split'
        )}
        if community_dim > 0:
            sylo_in.update({'community': Sylo(
                in_channels,
                community_dim,
                in_dim,
                rank=1,
                bias=False,
                symmetry=True,
                coupling='+'
            )})
        self.sylo1 = nn.ModuleDict(sylo_in)
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
        recombine = self._recombine
        norm_layer = self._norm_layer
        nlin = self._nlin

        layers = []
        layers.append(block(
            dim=dim,
            in_channels=in_channels,
            channels=channels,
            recombine=recombine,
            nlin=nlin,
            norm_layer=norm_layer,
            compression=compression
        ))
        for _ in range(1, blocks):
            layers.append(block(
                dim=dim,
                in_channels=channels,
                channels=channels,
                recombine=recombine,
                nlin=nlin,
                norm_layer=norm_layer,
                compression=None
            ))
        return nn.ModuleList(layers)

    def forward(self, x, query=None):
        x = [s(x) for s in self.sylo1.values()]
        x = torch.cat(x, -3)
        x = self.norm1(x)
        x = self.nlin(x)
        q_idx = 0
        for l in self.layers:
            for block in l:
                if query is None:
                    x = block(x)
                else:
                    x = block(x, query=query[q_idx:(q_idx + 2)])
                    q_idx += 2
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
                 community_dim=0,
                 potentials=None):
        super().__init__()
        init_arg = [{}]
        if init == 'svd':
            init_arg = [{'svd' : True}]
        elif init == 'resid':
            init_arg = [{'residualise' : True}]
        init_arg += [{} for _ in lattice_order_sequence[1:]]

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
        if potentials is not None:
            self.set_potentials(potentials)

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
            compressions=self._compressions,
            community_dim=community_dim
        )

    def set_potentials(self, potentials):
        self._compression_specs[0].set_potentials(potentials)
        try:
            for c in self._compressions:
                c.reset_parameters()
        except AttributeError:
            pass

    def forward(self, x, query=None):
        return self.model(x, query=query)
