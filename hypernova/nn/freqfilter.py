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
from itertools import chain
from ..functional import product_filtfilt
from ..functional.activation import amplitude_tanh
from ..init.iirfilter import iirfilter_init_, clamp_init_


class FrequencyDomainFilter(Module):
    """
    Filtering or convolution via transfer function multiplication in the
    frequency domain.

    Each time series in the input dataset is transformed into the frequency
    domain, where it is multiplied by the complex-valued transfer function of
    each filter in the module's bank. Each filtered frequency spectrum is then
    transformed back into the time domain. To ensure a zero-phase filter, the
    filtered time series are reversed and the process is repeated.

    Dimension
    ---------
    - Input: :math:`(N, *, C, T)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      C denotes number of variables or data channels, T denotes number of time
      points or observations.
    - Output: :math:`(N, *, F, C, T)`
      F denotes number of filters.

    Parameters
    ----------
    filter_specs : list(IIRFilterSpec)
        A list of filter specifications implemented as `IIRFilterSpec` objects
        (`hypernova.init.IIRFilterSpec`). These determine the filter bank that
        is applied to the input. Consult the `IIRFilterSpec` documentation for
        further details.
    dim : int or None
        Number of frequency bins. This must be conformant with the time series
        supplied as input. If you are uncertain about the dimension in the
        frequency domain, it is possible to instead provide the `time_dim`
        argument (the length of the time series), but either `time_dim` or
        `dim` (but not both) must be specified.
    time_dim : int or None
        Number of time points in the input time series. Either `time_dim` or
        `dim` (but not both) must be specified.
    filter : callable (default product_filtfilt)
        Callable function that takes as its arguments an input time series and
        a set of transfer functions. It transforms the input into the frequency
        domain, multiplies it by the transfer function bank, and transforms it
        back. By default, the `product_filtfilt` function is used to ensure a
        zero-phase filter.
    domain : 'linear' or 'atanh' (default 'atanh')
        Domain of the raw transfer function parameters. `linear` yields the
        untransformed transfer function, while `atanh` transforms the
        amplitudes of each bin by the inverse tanh (atanh) function. If the
        weights are in the `atanh` domain, then their amplitudes are
        transformed by the tanh function prior to convolution with the input.
        Using the `atanh` domain thereby constrains transfer function
        amplitudes to [0, 1) and prevents explosive gain.

    Attributes
    ----------
    weight : Tensor :math:`(F, D)`
        Filter bank transfer functions in the module's domain. F denotes the
        total number of filters in the bank, and D denotes the dimension of the
        input dataset in the frequency domain. The weights are initialised to
        emulate each  of the filters specified in the `filter_specs` parameter
        following the `iirfilter_init_` function.
    constrained_weight : Tensor :math:`(F, D)`
        The transfer function weights as seen by the input dataset in the
        frequency domain. This entails mapping the weights out of the `atanh`
        predomain and applying any clamps declared in the input specifications.
    clamp_points : Tensor :math:`(F, D)`
        Boolean-valued tensor mask indexing points in the transfer function
        that should be clamped to particular values. Any points so indexed will
        not be learnable. If this is None, then no clamp is applied.
    clamp_values : Tensor :math:`(V)`
        Tensor containing values to which the transfer functions are clamped. V
        denotes the total number of values to be clamped across all transfer
        functions. If this is None, then no clamp is applied.
    activation : callable
        Activation function that maps the internal module weights into the
        frequency domain of the data. If the module domain is `atanh`, then
        this is the `amplitude_tanh` function that passes the amplitude of each
        value through a tanh function while preserving its phase. If the domain
        is `linear` there is no activation function.
    """
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
        if self.clamp_points is not None:
            clamp_init_(self.clamp_points,
                        self.clamp_values,
                        self.filter_specs)

    def _set_dimension(self, dim, time_dim):
        if dim is None:
            if time_dim is None:
                raise ValueError('You must specify the dimension in either '
                                 'the frequency or time domain')
            else:
                dim = time_dim // 2 + 1
        return dim

    def _set_activation(self):
        if self.domain =='linear':
            return lambda x: x
        elif self.domain == 'atanh':
            return amplitude_tanh

    def _check_clamp(self):
        clamps = list(chain.from_iterable(
            [[len(f.keys()) for f in spec.clamps]
             for spec in self.filter_specs]))
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

    def __repr__(self):
        s = f'{self.__class__.__name__}(domain={self.domain}, filters=[\n'
        s += ',\n'.join([f'    {spec.__repr__()}'
                        for spec in self.filter_specs])
        s += '\n])'
        return s

    def forward(self, input):
        if input.size(0) > 1 and input.dim() > 1:
            input = input.unsqueeze(-3)
            weight = self.constrained_weight.unsqueeze(-2)
        else:
            weight = self.constrained_weight
        return self.filter(input, weight)
