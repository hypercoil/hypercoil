# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Covariance
~~~~~~~~~~
Modules supporting covariance estimation.
"""
import torch
from torch.nn import Module, Parameter, init


class _Cov(Module):
    def __init__(self, dim, estimator, max_lag, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, domain=('logit', 2)):
        super(_Cov, self).__init__()

        self.dim = dim
        self.estimator = estimator
        self.max_lag = max_lag
        self.out_channels = out_channels
        self.rowvar = rowvar
        self.bias = bias
        self.ddof = ddof
        self.l2 = l2
        self.noise = noise
        self.dropout = dropout
        self.domain = domain
        self.activation = self._set_activation()

        if self.max_lag is None:
            self.mask = None
            self.register_parameter('mask', None)

    def inject_noise(self, weight):
        if self.noise is not None:
            weight = self.noise.inject(weight)
        if self.dropout is not None:
            weight = self.dropout.inject(weight)
        return weight

    def train(self, mode=True):
        super(_Cov, self).train(mode)
        if self.noise is not None:
            self.noise.train(mode)
        if self.dropout is not None:
            self.dropout.train(mode)

    def eval(self):
        super(_Cov, self).eval()
        if self.noise is not None:
            self.noise.eval()
        if self.dropout is not None:
            self.dropout.eval()

    @property
    def weight(self):
        preweight = self.activation(self.preweight)
        return self.inject_noise(preweight)

    def _set_activation(self):
        if self.domain == 'identity':
            return lambda x: x
        func, scale = self.domain
        if func =='linear':
            return lambda x: scale * x
        elif func == 'logit':
            return lambda x: scale * torch.sigmoid(x)
        elif func == 'atanh':
            return lambda x: scale * torch.tanh(x)


class _UnaryCov(_Cov):
    def __init__(self, dim, estimator, max_lag, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, domain=('logit', 2)):
        super(_UnaryCov, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            domain=domain, out_channels=out_channels
        )

    def forward(self, input):
        if self.out_channels > 1 and input.dim() > 2 and input.size(-3) > 1:
            input = input.unsqueeze(-3)
        return self.estimator(
            input,
            rowvar=self.rowvar,
            bias=self.bias,
            ddof=self.ddof,
            weight=self.weight,
            l2=self.l2
        )


class UnaryCovarianceUW(_UnaryCov):
    def __init__(self, dim, estimator, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None):
        super(UnaryCovarianceUW, self).__init__(
            dim=dim, estimator=estimator, max_lag=0, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            domain='identity', out_channels=out_channels
        )
        self.preweight = Parameter(torch.Tensor(
            self.out_channels, self.dim, self.dim
        ), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.preweight[:] = torch.eye(self.dim)
