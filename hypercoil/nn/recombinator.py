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
from ..functional.matrix import expand_outer


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
                 bias=True, positive_only=False, init=None,
                 device=None, dtype=None):
        super(Recombinator, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}

        if init is None:
            init = {'nonlinearity': 'linear'}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.positive_only = positive_only
        self.init = init

        self.weight = Parameter(torch.empty(
            out_channels, in_channels, **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(
                out_channels, **factory_kwargs))
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
            with torch.no_grad():
                self.weight.abs_()

    def extra_repr(self):
        s = 'in_channels={}, out_channels={}'.format(
            self.in_channels, self.out_channels)
        if self.bias is None:
            s += ', bias=False'
        return s

    def forward(self, input, query=None):
        return recombine(
            input=input,
            mixture=self.weight,
            bias=self.bias,
            query=query
        )


# TODO: Move to functional if we decide to keep this instead of just using 1x1
def recombine(input, mixture, query=None,
              query_L=None, query_R=None, bias=None):
    """
    Create a new mixture of the input feature maps.

    Parameters
    ----------
    input: Tensor (N x C_in x H x W)
        Stack of input matrices or feature maps.
    mixture: Tensor (C_out x C_in)
        Mixture matrix or recombinator.
    query: Tensor (N x C_in x C_in)
        If provided, the mixture is recomputed as the dot product similarity
        between each mixture vector and each query vector, and the softmax of
        the result is used to form convex combinations of inputs.
    bias: Tensor (C_in)
        Bias term to apply after recombining.
    """
    if query_L is not None:
        mixture = mixture @ query_L
        if query_R is None:
            query_R = query_L
        mixture = mixture @ query_R.transpose(-1, -2)
        mixture = torch.softmax(mixture, -1).unsqueeze(-3)
    if query is not None:
        mixture = mixture @ query
        mixture = torch.softmax(mixture, -1).unsqueeze(-3)
    output = (mixture @ input.transpose(1, 2)).transpose(1, 2)
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    return output


class QueryEncoder(nn.Module):
    """
    Query encoder for recombinators.
    """
    def __init__(self, num_embeddings, embedding_dim, query_dim,
                 common_layer_dim, specific_layer_dim, nlin=None,
                 progressive_specificity=False, rank='full', noise_dim=0):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim
        )
        self.noise_dim = noise_dim
        self.rank = rank
        self.nlin = nlin or torch.nn.ReLU(inplace=True)
        self.progressive_specificity = progressive_specificity
        common_dims = [embedding_dim + noise_dim, *common_layer_dim]
        specific_in_dim = common_dims[-1]
        self.common_layers = torch.nn.ModuleList([
            torch.nn.Linear(i, o) for i, o
            in zip(common_dims[:-1], common_dims[1:])
        ])
        specific_layers = []
        query_layers = []
        for q_dim in query_dim:
            specific_dims = [specific_in_dim, *specific_layer_dim]
            s_layers = [
                torch.nn.Linear(i, o) for i, o
                in zip(specific_dims[:-1], specific_dims[1:])
            ]
            q_layers = self._cfg_query_module(q_dim, specific_dims[-1])
            if self.progressive_specificity:
                specific_in_dim = specific_dims[-1]
            specific_layers += [torch.nn.ModuleList(s_layers)]
            query_layers += [torch.nn.ModuleList(q_layers)]
        self.specific_layers = torch.nn.ModuleList(specific_layers)
        self.query_layers = torch.nn.ModuleList(query_layers)

    def reset_parameters(self):
        with torch.no_grad():
            torch.nn.init.normal_(self.embedding.weight)

    def _cfg_query_module(self, query_dim, in_dim):
        if self.rank == 'full':
            return [torch.nn.Linear(in_dim, (query_dim ** 2))]
        else:
            return [
                torch.nn.Linear(in_dim, query_dim * self.rank),
                torch.nn.Linear(in_dim, query_dim * self.rank)
            ]

    def _conform_query_dim(self, q, dim):
        return q.view(*q.shape[:-1], -1, dim)

    def forward(self, x, skip_embedding=False, embedding_only=False):
        if not skip_embedding:
            x = self.embedding(x)
        if self.noise_dim > 0:
            noise = torch.randn(*x.shape[:-1], self.noise_dim)
            x = torch.cat((x, noise), -1)
        e = x
        if embedding_only:
            return e
        for l in self.common_layers:
            x = l(x)
            x = self.nlin(x)
        queries = []
        for query_axis, query_out in zip(self.specific_layers, self.query_layers):
            q = x
            for l in query_axis:
                q = l(q)
                q = self.nlin(q)
            if self.progressive_specificity:
                x = q
            if self.rank == 'full':
                q = query_out[0](q)
                q_dim = torch.sqrt(q.shape[-1].long())
                q = self._conform_query_dim(q, q_dim)
            else:
                q_L, q_R = [l(q) for l in query_out]
                q_L = self._conform_query_dim(q_L, self.rank)
                q_R = self._conform_query_dim(q_R, self.rank)
                q = expand_outer(L=q_L, R=q_R)
            queries += [q]
        return tuple(queries), e
