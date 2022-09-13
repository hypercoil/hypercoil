# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Vertical compression layer.
"""
import torch
from torch import nn
from torch.nn import Parameter
from ..init.mpbl import BipartiteLatticeInit
from ..functional.sylo import vertical_compression


class VerticalCompression(nn.Module):
    r"""
    Compress a graph by fusing vertices. For an adjacency matrix A, this
    layer applies the transform
    :math:`\left(C_{row} A\right) \times C_{col}^{\intercal}` so that an
    :math:`H_{in} \times W_{in}` matrix is mapped to an
    :math:`H_{out} \times W_{out}` matrix.
    """
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, init, renormalise=True,
                 fold_channels=True, learnable=True, device=None, dtype=None,
                 forward_operation='compress'):
        super(VerticalCompression, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.factory_kwargs = factory_kwargs
        self.in_features = in_features
        self.out_features = out_features
        self.initialised = False
        try:
            if isinstance(init[0], BipartiteLatticeInit):
                pass
        except TypeError:
            init = [init]
        self._out_channels = [(i.n_lattices * i.channel_multiplier)
                              for i in init]
        self.out_channels = sum(self._out_channels)
        self.C = Parameter(torch.empty(
            (self.out_channels, self.out_features, self.in_features),
            **factory_kwargs))
        self.mask = Parameter(torch.empty(
            (self.out_channels, self.out_features, self.in_features),
            dtype=torch.bool, device=device
        ), requires_grad=False)
        self.sign = Parameter(torch.empty(
            (self.out_channels, 1, 1), **factory_kwargs
        ), requires_grad=False)
        if not learnable:
            self.C.requires_grad = False
        self.init = init
        self.renormalise = renormalise
        self.fold_channels = fold_channels
        self.forward_operation = forward_operation
        self.reset_parameters()

    def reset_parameters(self):
        try:
            factory_kwargs = self.factory_kwargs
            start = 0
            for init, n_ch in zip(self.init, self._out_channels):
                end = start + n_ch
                tsr, msk = (
                    torch.empty((n_ch, self.out_features, self.in_features),
                                **factory_kwargs),
                    torch.empty((n_ch, self.out_features, self.in_features),
                                dtype=torch.bool,
                                device=factory_kwargs['device'])
                )
                init(tsr, msk)
                with torch.no_grad():
                    self.C[start:end] = tsr
                    self.mask[start:end] = msk
                    self.sign[start:end] = init.sign_vector()
                start = end
            self.initialised = True
        except AttributeError:
            return

        self.sparsity = (torch.sum(self.mask) /
                         torch.numel(self.mask)).item()

    def extra_repr(self):
        if self.initialised:
            s = ('compression=({}, {}), out_channels={}, sparsity={:.4}'
                ).format(self.in_features, self.out_features,
                         self.out_channels, self.sparsity)
        else:
            s = ('uninitialised=True, compression=({}, {}), out_channels={}'
                ).format(self.in_features, self.out_features,
                         self.out_channels)
        return s

    def reconstruct(self, compressed):
        if not self.initialised:
            raise ValueError(
                'Vertical compression module has not been initialised. '
                'Ensure that potentials have been configured for all '
                'initialisers passed to the module.'
            )
        return vertical_compression(
            input=compressed,
            row_compressor=(self.mask * self.C).transpose(-1, -2),
            renormalise=self.renormalise,
            remove_diagonal=True,
            fold_channels=self.fold_channels,
            sign=self.sign
        )

    def compress(self, input):
        if not self.initialised:
            raise ValueError(
                'Vertical compression module has not been initialised. '
                'Ensure that potentials have been configured for all '
                'initialisers passed to the module.'
            )
        return vertical_compression(
            input=input,
            row_compressor=(self.mask * self.C),
            renormalise=self.renormalise,
            remove_diagonal=True,
            fold_channels=self.fold_channels,
            sign=self.sign
        )

    def forward(self, input):
        if self.forward_operation == 'compress':
            return self.compress(input=input)
        elif self.forward_operation == 'reconstruct':
            return self.reconstruct(compressed=input)
