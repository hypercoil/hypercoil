# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Frequency-domain filter
~~~~~~~~~~~~~~~~~~~~~~~
Modules supporting filtering/convolution as a product in the frequency domain.
"""
import math
import torch
from torch.nn import Module, Parameter, init
from ..functional import product_filtfilt
from ..functional.activation import amplitude_tanh
from ..init.iirfilter import iirfilter_init_, clamp_init_


class FrequencyDomainFilter(Module):
    def __init__(self, filter_specs, dim=None, time_dim=None,
                 filter=product_filtfilt, domain='atanh'):
        super(FrequencyDomainFilter, self).__init__()

        self.filter_specs = filter_specs
        self.dim = self._set_dimension(dim, time_dim)
        self.channels = sum([spec.n_filters for spec in self.filter_specs])
        self.filter = filter
        self.domain = domain
        self.activation = self._set_activation()
        self.clamp_points, self.clamp_values = self._check_clamp()

        self.weight = Parameter(torch.complex(
            torch.Tensor(self.channels, self.dim),
            torch.Tensor(self.channels, self.dim)
        ))

        self.reset_parameters()

    def reset_parameters(self):
        iirfilter_init_(self.weight, self.filter_specs, domain=self.domain)
        clamp_init_(self.clamp_points, self.clamp_values, self.filter_specs)

    def _set_dimension(self, dim, time_dim):
        if dim is None:
            if time_dim is None:
                raise ValueError('You must specify the dimension in either '
                                 'the frequency or time domain')
            else:
                dim = (time_dim + 1) // 2
        return dim

    def _set_activation(self):
        if self.domain =='linear':
            return lambda x: x
        elif self.domain == 'atanh':
            return amplitude_tanh

    def _check_clamp(self):
        clamps = [[len(f.keys()) for f in spec.clamps]
                  for spec in self.filter_specs]
        n_clamps = int(torch.Tensor(clamps).sum().item())
        if n_clamps <= 0:
            self.register_parameter('clamp_points', None)
            self.register_parameter('clamp_values', None)
            return None, None
        clamp_points = Parameter(
            torch.Tensor(self.channels, self.dim).bool(),
            requires_grad=False)
        clamp_values = Parameter(torch.complex(
            torch.Tensor(n_clamps),
            torch.Tensor(n_clamps)),
            requires_grad=False)
        return clamp_points, clamp_values

    def _apply_clamps(self, weight):
        if self.clamp_points is not None:
            weight[self.clamp_points] = self.clamp_values
        return weight

    @property
    def constrained_weight(self):
        return self._apply_clamps(self.activation(self.weight))

    def forward(self, input):
        return self.filter(input, self.constrained_weight)

    def __repr__(self):
        s = f'{self.__class__.__name__}(domain={self.domain}, [\n'
        for spec in self.filter_specs:
            s += '    '
            s += spec.__repr__()
            s += '\n'
        s += '])'
        return s
