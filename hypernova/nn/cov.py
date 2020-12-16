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
from ..functional.activation import laplace
from ..functional.domain import Identity
from ..functional.matrix import toeplitz
from ..init.laplace import laplace_init_
from ..init.toeplitz import toeplitz_init_


class _Cov(Module):
    def __init__(self, dim, estimator, max_lag, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, domain=None):
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
        self.domain = domain or Identity()

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
        return self.domain.image(self.preweight)

    @property
    def postweight(self):
        return self.inject_noise(self.weight)


class _UnaryCov(_Cov):
    def forward(self, input):
        if input.dim() > 2 and self.out_channels > 1 and input.size(-3) > 1:
            input = input.unsqueeze(-3)
        return self.estimator(
            input,
            rowvar=self.rowvar,
            bias=self.bias,
            ddof=self.ddof,
            weight=self.postweight,
            l2=self.l2
        )


class _BinaryCov(_Cov):
    def forward(self, x, y):
        return self.estimator(
            x, y,
            rowvar=self.rowvar,
            bias=self.bias,
            ddof=self.ddof,
            weight=self.postweight,
            l2=self.l2
        )


class _WeightedCov(_Cov):
    def __init__(self, dim, estimator, max_lag, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, domain=None):
        super(_WeightedCov, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            domain=domain, out_channels=out_channels
        )
        self.preweight = Parameter(torch.Tensor(
            self.out_channels, self.dim, self.dim
        ))
        self.mask = Parameter(torch.Tensor(
            self.dim, self.dim
        ).bool(), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        toeplitz_init_(
            self.mask,
            torch.Tensor([1 for _ in range(self.max_lag + 1)])
        )
        toeplitz_init_(
            self.preweight,
            laplace(torch.arange(self.max_lag + 1))
        )

    @property
    def postweight(self):
        return self.inject_noise(self.weight) * self.mask


class _ToeplitzWeightedCov(_Cov):
    def __init__(self, dim, estimator, max_lag=1, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, domain=None):
        super(_ToeplitzWeightedCov, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            domain=domain, out_channels=out_channels
        )
        self.prepreweight_c = Parameter(torch.Tensor(
            self.max_lag + 1, self.out_channels
        ))
        self.prepreweight_r = Parameter(torch.Tensor(
            self.max_lag + 1, self.out_channels
        ))
        self.reset_parameters()

    def reset_parameters(self):
        laplace_init_(self.prepreweight_c, loc=(0, 0), excl_axis=[1])
        laplace_init_(self.prepreweight_r, loc=(0, 0), excl_axis=[1])

    @property
    def preweight(self):
        return toeplitz(c=self.prepreweight_c,
                        r=self.prepreweight_r,
                        dim=(self.dim, self.dim))


class _UnweightedCov(_Cov):
    def __init__(self, dim, estimator, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None):
        super(_UnweightedCov, self).__init__(
            dim=dim, estimator=estimator, max_lag=0, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            domain=None, out_channels=out_channels
        )
        self.preweight = Parameter(torch.Tensor(
            self.out_channels, self.dim, self.dim
        ), requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        self.preweight[:] = torch.eye(self.dim)


class UnaryCovariance(_UnaryCov, _WeightedCov):
    def __init__(self, dim, estimator, max_lag=1, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, domain=None):
        super(UnaryCovariance, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            domain=domain, out_channels=out_channels
        )


class UnaryCovarianceTW(_UnaryCov, _ToeplitzWeightedCov):
    def __init__(self, dim, estimator, max_lag=1, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, domain=None):
        super(UnaryCovarianceTW, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            domain=domain, out_channels=out_channels
        )


class UnaryCovarianceUW(_UnaryCov, _UnweightedCov):
    def __init__(self, dim, estimator, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None):
        super(UnaryCovarianceUW, self).__init__(
            dim=dim, estimator=estimator, max_lag=0, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            domain=None, out_channels=out_channels
        )


class BinaryCovariance(_BinaryCov, _WeightedCov):
    def __init__(self, dim, estimator, max_lag=1, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, domain=None):
        super(BinaryCovariance, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            domain=domain, out_channels=out_channels
        )


class BinaryCovarianceTW(_BinaryCov, _ToeplitzWeightedCov):
    def __init__(self, dim, estimator, max_lag=1, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, domain=None):
        super(BinaryCovarianceTW, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            domain=domain, out_channels=out_channels
        )


class BinaryCovarianceUW(_BinaryCov, _UnweightedCov):
    def __init__(self, dim, estimator, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None):
        super(BinaryCovarianceUW, self).__init__(
            dim=dim, estimator=estimator, max_lag=0, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            domain=None, out_channels=out_channels
        )
