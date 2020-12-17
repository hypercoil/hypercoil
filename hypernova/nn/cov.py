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
    """
    Base class for modules that estimate covariance or derived measures.
    """
    def __init__(self, dim, estimator, max_lag=0, out_channels=1,
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
    """
    Base class for covariance estimators that operate on a single variable
    tensor.
    """
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
    """
    Base class for covariance estimators that operate on a pair of variable
    tensors.
    """
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
    """
    Base class for covariance estimators with a full complement of learnable
    weights.
    """
    def __init__(self, dim, estimator, max_lag=0, out_channels=1,
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
        if self.max_lag is not None:
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
        if self.mask is not None:
            return self.inject_noise(self.weight) * self.mask
        return self.inject_noise(self.weight)


class _ToeplitzWeightedCov(_Cov):
    """
    Base class for covariance estimators with a single learnable weight for
    each time lag.
    """
    def __init__(self, dim, estimator, max_lag=0, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, domain=None):
        super(_ToeplitzWeightedCov, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            domain=domain, out_channels=out_channels
        )
        if self.max_lag is not None:
            self.mask = Parameter(torch.Tensor(
                self.dim, self.dim
            ).bool(), requires_grad=False)
        self.prepreweight_c = Parameter(torch.Tensor(
            self.max_lag + 1, self.out_channels
        ))
        self.prepreweight_r = Parameter(torch.Tensor(
            self.max_lag + 1, self.out_channels
        ))
        self.reset_parameters()

    def reset_parameters(self):
        toeplitz_init_(
            self.mask,
            torch.Tensor([1 for _ in range(self.max_lag + 1)])
        )
        laplace_init_(self.prepreweight_c, loc=(0, 0), excl_axis=[1],
                      var=0, domain=self.domain)
        laplace_init_(self.prepreweight_r, loc=(0, 0), excl_axis=[1],
                      var=0, domain=self.domain)

    @property
    def preweight(self):
        return toeplitz(c=self.prepreweight_c,
                        r=self.prepreweight_r,
                        dim=(self.dim, self.dim),
                        fill_value=self.domain.preimage(torch.zeros(1)).item())

    @property
    def postweight(self):
        if self.mask is not None:
            return self.inject_noise(self.weight) * self.mask
        return self.inject_noise(self.weight)


class _UnweightedCov(_Cov):
    """
    Base class for covariance estimators without learnable parameters.
    """
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
    """
    Covariance measures of a single tensor, with a full complement of learnable
    weights.

    The input tensor is interpreted as a set of multivariate observations.
    A covariance estimator computes some measure of statistical dependence
    among the variables in each observation, with the potential addition of
    stochastic noise and dropout to re-weight observations and regularise the
    model.

    Dimension
    ---------
    - Input: :math:`(N, *, C, O)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      C denotes number of variables or data channels, O denotes number of time
      points or observations.
    - Output: :math:`(N, *, W, C, C)`
      W denotes number of sets of weights.

    Parameters
    ----------
    dim : int
        Number of observations `O` per data instance. This determines the
        dimension of each slice of the covariance weight tensor.
    estimator : callable
        Covariance estimator, e.g. from `hypernova.functional.cov`. The
        estimator must be unary: it should accept a single tensor rather than
        multiple tensors. Some available options are:
        - `cov`: Raw empirical covariance.
        - `corr`: Pearson correlation.
        - `precision`: Precision.
        - `partialcorr`: Partial correlation.
    max_lag : int or None (default 0)
        Maximum lag to include in the weight matrix. If this is not None, the
        structure of the weight matrix is constrained to allow nonzero entries
        only along diagonals that are a maximum offset of `max_lag` from the
        main diagonal. The default value of 0 permits weights only along the
        main diagonal.
    out_channels : int (default 1)
        Number of weight sets to include. For each weight set, the module
        produces an output channel.
    rowvar : bool (default True)
        Indicates that the last axis of the input tensor is the observation
        axis and the penultimate axis is the variable axis. If False, then this
        relationship is transposed.
    bias : bool (default False)
        Indicates that the biased normalisation (i.e., division by `N` in the
        unweighted case) should be performed. By default, normalisation of the
        covariance is unbiased (i.e., division by `N - 1`).
    ddof : int or None (default None)
        Degrees of freedom for normalisation. If this is specified, it
        overrides the normalisation factor automatically determined using the
        `bias` parameter.
    l2: nonnegative float (default 0)
        L2 regularisation term to add to the maximum likelihood estimate of the
        covariance matrix. This can be set to a positive value to obtain an
        intermediate for estimating the regularised inverse covariance or to
        ensure that the covariance matrix is non-singular (if, for instance,
        it needs to be inverted or projected into a tangent space).
    noise: NoiseSource object or None (default None)
        Noise source to inject into the weights. A diagonal noise source adds
        stochasticity into observation weights and can be used to regularise a
        zero-lag covariance. A positive semidefinite noise source ensures that
        the weight tensor remains in the positive semidefinite cone.
    dropout: DropoutSource object or None (default None)
        Dropout source to inject into the weights. A dropout source can be used
        to randomly ignore a subset of observations and thereby perform a type
        of data augmentation (similar to bootstrapped covariance estimates).
    domain: Domain object or None (default None)
        A domain object from `hypernova.functional.domain`, used to specify the
        domain of the weights. An `Identity` object yields the raw weights,
        while objects such as `Logit` domains transform the weights prior to
        multiplication with the input. Using alternative domains can thereby
        constrain the weights to some desirable interval (e.g., [0, +f)).

    Attributes
    ----------
    mask : Tensor :math:`(W, O, O)`
        Boolean-valued tensor indicating the entries of the weight tensor that
        are permitted to take nonzero values. This is determined by the
        specified `max_lag` parameter at initialisation.
    preweight : Tensor :math:`(W, O, O)`
        Tensor containing raw internal values of the weights. This is the
        preimage of the weights under the transformation specified in the
        `domain` object, prior to the enforcement of any symmetry constraints.
        The preweight is initialised as the preimage of a Toeplitz banded
        matrix where the weight of each diagonal is set according to a double
        exponential function with a maximum of 1 at the origin (zero lag).
    weight : Tensor :math:`(W, O, O)`
        Tensor containing importance or coupling weights for the observations.
        If this tensor is 1-dimensional, each entry weights the corresponding
        observation in the covariance computation. If it is 2-dimensional,
        then it must be square, symmetric, and positive semidefinite. In this
        case, diagonal entries again correspond to relative importances, while
        off-diagonal entries indicate coupling factors. For instance, a banded
        or multi-diagonal tensor can be used to specify inter-temporal coupling
        for a time series covariance.
    postweight : Tensor :math:`(W, O, O)`
        Final weights as seen by the data. This is the weight after any noise
        and dropout sources are applied and after final masking.
    """
    def __init__(self, dim, estimator, max_lag=0, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, domain=None):
        super(UnaryCovariance, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            domain=domain, out_channels=out_channels
        )


class UnaryCovarianceTW(_UnaryCov, _ToeplitzWeightedCov):
    """
    Covariance measures of a single tensor, with a single learnable weight for
    each time lag.

    The input tensor is interpreted as a set of multivariate observations.
    A covariance estimator computes some measure of statistical dependence
    among the variables in each observation, with the potential addition of
    stochastic noise and dropout to re-weight observations and regularise the
    model.

    Dimension
    ---------
    - Input: :math:`(N, *, C, O)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      C denotes number of variables or data channels, O denotes number of time
      points or observations.
    - Output: :math:`(N, *, W, C, C)`
      W denotes number of sets of weights.

    Parameters
    ----------
    dim : int
        Number of observations `O` per data instance. This determines the
        dimension of each slice of the covariance weight tensor.
    estimator : callable
        Covariance estimator, e.g. from `hypernova.functional.cov`. The
        estimator must be unary: it should accept a single tensor rather than
        multiple tensors. Some available options are:
        - `cov`: Raw empirical covariance.
        - `corr`: Pearson correlation.
        - `precision`: Precision.
        - `partialcorr`: Partial correlation.
    max_lag : int or None (default 0)
        Maximum lag to include in the weight matrix. If this is not None, the
        structure of the weight matrix is constrained to allow nonzero entries
        only along diagonals that are a maximum offset of `max_lag` from the
        main diagonal. The default value of 0 permits weights only along the
        main diagonal.
    out_channels : int (default 1)
        Number of weight sets to include. For each weight set, the module
        produces an output channel.
    rowvar : bool (default True)
        Indicates that the last axis of the input tensor is the observation
        axis and the penultimate axis is the variable axis. If False, then this
        relationship is transposed.
    bias : bool (default False)
        Indicates that the biased normalisation (i.e., division by `N` in the
        unweighted case) should be performed. By default, normalisation of the
        covariance is unbiased (i.e., division by `N - 1`).
    ddof : int or None (default None)
        Degrees of freedom for normalisation. If this is specified, it
        overrides the normalisation factor automatically determined using the
        `bias` parameter.
    l2: nonnegative float (default 0)
        L2 regularisation term to add to the maximum likelihood estimate of the
        covariance matrix. This can be set to a positive value to obtain an
        intermediate for estimating the regularised inverse covariance or to
        ensure that the covariance matrix is non-singular (if, for instance,
        it needs to be inverted or projected into a tangent space).
    noise: NoiseSource object or None (default None)
        Noise source to inject into the weights. A diagonal noise source adds
        stochasticity into observation weights and can be used to regularise a
        zero-lag covariance. A positive semidefinite noise source ensures that
        the weight tensor remains in the positive semidefinite cone.
    dropout: DropoutSource object or None (default None)
        Dropout source to inject into the weights. A dropout source can be used
        to randomly ignore a subset of observations and thereby perform a type
        of data augmentation (similar to bootstrapped covariance estimates).
    domain: Domain object or None (default None)
        A domain object from `hypernova.functional.domain`, used to specify the
        domain of the weights. An `Identity` object yields the raw weights,
        while objects such as `Logit` domains transform the weights prior to
        multiplication with the input. Using alternative domains can thereby
        constrain the weights to some desirable interval (e.g., [0, +f)).

    Attributes
    ----------
    prepreweight_c, prepreweight_r : Tensor :math:`(W, L)`
        Toeplitz matrix generators for the columns (lag) and rows (lead) of the
        weight matrix. L denotes the maximum lag. These parameters are injected
        along each diagonal of the weight matrix up to the maximum lag. The
        prepreweights are initialised as double exponentials with a maximum of
        1 at the origin (zero lag).
    preweight : Tensor :math:`(W, O, O)`
        Tensor containing raw internal values of the weights. This is the
        preimage of the weights under the transformation specified in the
        `domain` object, prior to the enforcement of any symmetry constraints.
    weight : Tensor :math:`(W, O, O)`
        Tensor containing importance or coupling weights for the observations.
        If this tensor is 1-dimensional, each entry weights the corresponding
        observation in the covariance computation. If it is 2-dimensional,
        then it must be square, symmetric, and positive semidefinite. In this
        case, diagonal entries again correspond to relative importances, while
        off-diagonal entries indicate coupling factors. For instance, a banded
        or multi-diagonal tensor can be used to specify inter-temporal coupling
        for a time series covariance.
    postweight : Tensor :math:`(W, O, O)`
        Final weights as seen by the data. This is the weight after any noise
        and dropout sources are applied and after final masking.
    """
    def __init__(self, dim, estimator, max_lag=0, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, domain=None):
        super(UnaryCovarianceTW, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            domain=domain, out_channels=out_channels
        )


class UnaryCovarianceUW(_UnaryCov, _UnweightedCov):
    """
    Covariance measures of a single tensor, without learnable weights.

    The input tensor is interpreted as a set of multivariate observations.
    A covariance estimator computes some measure of statistical dependence
    among the variables in each observation, with the potential addition of
    stochastic noise and dropout to re-weight observations and regularise the
    model.

    Though it does not contain learnable weights, this module nonetheless
    supports multiple weight channels for the purpose of data augmentation. By
    injecting the weights in different data channels with different noise and
    dropout terms, it can produce several different estimates of covariance
    from the same source data.

    Dimension
    ---------
    - Input: :math:`(N, *, C, O)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      C denotes number of variables or data channels, O denotes number of time
      points or observations.
    - Output: :math:`(N, *, W, C, C)`
      W denotes number of sets of weights.

    Parameters
    ----------
    dim : int
        Number of observations `O` per data instance. This determines the
        dimension of each slice of the covariance weight tensor.
    estimator : callable
        Covariance estimator, e.g. from `hypernova.functional.cov`. The
        estimator must be unary: it should accept a single tensor rather than
        multiple tensors. Some available options are:
        - `cov`: Raw empirical covariance.
        - `corr`: Pearson correlation.
        - `precision`: Precision.
        - `partialcorr`: Partial correlation.
    out_channels : int (default 1)
        Number of weight sets to include. For each weight set, the module
        produces an output channel.
    rowvar : bool (default True)
        Indicates that the last axis of the input tensor is the observation
        axis and the penultimate axis is the variable axis. If False, then this
        relationship is transposed.
    bias : bool (default False)
        Indicates that the biased normalisation (i.e., division by `N` in the
        unweighted case) should be performed. By default, normalisation of the
        covariance is unbiased (i.e., division by `N - 1`).
    ddof : int or None (default None)
        Degrees of freedom for normalisation. If this is specified, it
        overrides the normalisation factor automatically determined using the
        `bias` parameter.
    l2: nonnegative float (default 0)
        L2 regularisation term to add to the maximum likelihood estimate of the
        covariance matrix. This can be set to a positive value to obtain an
        intermediate for estimating the regularised inverse covariance or to
        ensure that the covariance matrix is non-singular (if, for instance,
        it needs to be inverted or projected into a tangent space).
    noise: NoiseSource object or None (default None)
        Noise source to inject into the weights. A diagonal noise source adds
        stochasticity into observation weights and can be used to regularise a
        zero-lag covariance. A positive semidefinite noise source ensures that
        the weight tensor remains in the positive semidefinite cone.
    dropout: DropoutSource object or None (default None)
        Dropout source to inject into the weights. A dropout source can be used
        to randomly ignore a subset of observations and thereby perform a type
        of data augmentation (similar to bootstrapped covariance estimates).

    Attributes
    ----------
    weight : Tensor :math:`(W, O, O)`
        Tensor containing importance or coupling weights for the observations.
        If this tensor is 1-dimensional, each entry weights the corresponding
        observation in the covariance computation. If it is 2-dimensional,
        then it must be square, symmetric, and positive semidefinite. In this
        case, diagonal entries again correspond to relative importances, while
        off-diagonal entries indicate coupling factors. For instance, a banded
        or multi-diagonal tensor can be used to specify inter-temporal coupling
        for a time series covariance.
    postweight : Tensor :math:`(W, O, O)`
        Final weights as seen by the data. This is the weight after any noise
        and dropout sources are applied and after final masking.
    """
    def __init__(self, dim, estimator, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None):
        super(UnaryCovarianceUW, self).__init__(
            dim=dim, estimator=estimator, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            out_channels=out_channels
        )


class BinaryCovariance(_BinaryCov, _WeightedCov):
    """
    Covariance measures using variables stored in two tensors, with a full
    complement of learnable weights.

    The input tensors are interpreted as sets of multivariate observations.
    A covariance estimator computes some measure of statistical dependence
    among the variables in each observation, with the potential addition of
    stochastic noise and dropout to re-weight observations and regularise the
    model.

    Dimension
    ---------
    - Input 1: :math:`(N, *, C_1, O)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      :math:`C_1` denotes number of variables or data channels, O denotes
      number of time points or observations.
    - Input 2: :math:`(N, *, C_2, O)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      :math:`C_2` denotes number of variables or data channels, O denotes
      number of time points or observations.
    - Output: :math:`(N, *, W, C_{*}, C_{*})`
      W denotes number of sets of weights. :math:`C_{*}` can denote either
      :math:`C_1` or :math:`C_2`, depending on the estimator provided. Paired
      estimators produce one axis of each size, while conditional estimators
      produce both axes of size :math:`C_1`.

    Parameters
    ----------
    dim : int
        Number of observations `O` per data instance. This determines the
        dimension of each slice of the covariance weight tensor.
    estimator : callable
        Covariance estimator, e.g. from `hypernova.functional.cov`. The
        estimator must be binary: it should accept two tensors rather than one.
        Some available options are:
        - `pairedcov`: Empirical covariance between variables in tensor 1 and
          those in tensor 2.
        - `pairedcorr`: Pearson correlation between variables in tensor 1 and
          those in tensor 2.
        - `conditionalcov`: Covariance between variables in tensor 1 after
          conditioning on variables in tensor 2. Can be used to control for the
          effects of confounds and is equivalent to confound regression with
          the addition of an intercept term.
        - `conditionalcorr`: Pearson correlation between variables in tensor 1
          after conditioning on variables in tensor 2.
    max_lag : int or None (default 0)
        Maximum lag to include in the weight matrix. If this is not None, the
        structure of the weight matrix is constrained to allow nonzero entries
        only along diagonals that are a maximum offset of `max_lag` from the
        main diagonal. The default value of 0 permits weights only along the
        main diagonal.
    out_channels : int (default 1)
        Number of weight sets to include. For each weight set, the module
        produces an output channel.
    rowvar : bool (default True)
        Indicates that the last axis of the input tensor is the observation
        axis and the penultimate axis is the variable axis. If False, then this
        relationship is transposed.
    bias : bool (default False)
        Indicates that the biased normalisation (i.e., division by `N` in the
        unweighted case) should be performed. By default, normalisation of the
        covariance is unbiased (i.e., division by `N - 1`).
    ddof : int or None (default None)
        Degrees of freedom for normalisation. If this is specified, it
        overrides the normalisation factor automatically determined using the
        `bias` parameter.
    l2: nonnegative float (default 0)
        L2 regularisation term to add to the maximum likelihood estimate of the
        covariance matrix. This can be set to a positive value to obtain an
        intermediate for estimating the regularised inverse covariance or to
        ensure that the covariance matrix is non-singular (if, for instance,
        it needs to be inverted or projected into a tangent space).
    noise: NoiseSource object or None (default None)
        Noise source to inject into the weights. A diagonal noise source adds
        stochasticity into observation weights and can be used to regularise a
        zero-lag covariance. A positive semidefinite noise source ensures that
        the weight tensor remains in the positive semidefinite cone.
    dropout: DropoutSource object or None (default None)
        Dropout source to inject into the weights. A dropout source can be used
        to randomly ignore a subset of observations and thereby perform a type
        of data augmentation (similar to bootstrapped covariance estimates).
    domain: Domain object or None (default None)
        A domain object from `hypernova.functional.domain`, used to specify the
        domain of the weights. An `Identity` object yields the raw weights,
        while objects such as `Logit` domains transform the weights prior to
        multiplication with the input. Using alternative domains can thereby
        constrain the weights to some desirable interval (e.g., [0, +f)).

    Attributes
    ----------
    mask : Tensor :math:`(W, O, O)`
        Boolean-valued tensor indicating the entries of the weight tensor that
        are permitted to take nonzero values. This is determined by the
        specified `max_lag` parameter at initialisation.
    preweight : Tensor :math:`(W, O, O)`
        Tensor containing raw internal values of the weights. This is the
        preimage of the weights under the transformation specified in the
        `domain` object, prior to the enforcement of any symmetry constraints.
        The preweight is initialised as the preimage of a Toeplitz banded
        matrix where the weight of each diagonal is set according to a double
        exponential function with a maximum of 1 at the origin (zero lag).
    weight : Tensor :math:`(W, O, O)`
        Tensor containing importance or coupling weights for the observations.
        If this tensor is 1-dimensional, each entry weights the corresponding
        observation in the covariance computation. If it is 2-dimensional,
        then it must be square, symmetric, and positive semidefinite. In this
        case, diagonal entries again correspond to relative importances, while
        off-diagonal entries indicate coupling factors. For instance, a banded
        or multi-diagonal tensor can be used to specify inter-temporal coupling
        for a time series covariance.
    postweight : Tensor :math:`(W, O, O)`
        Final weights as seen by the data. This is the weight after any noise
        and dropout sources are applied and after final masking.
    """
    def __init__(self, dim, estimator, max_lag=0, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, domain=None):
        super(BinaryCovariance, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            domain=domain, out_channels=out_channels
        )


class BinaryCovarianceTW(_BinaryCov, _ToeplitzWeightedCov):
    """
    Covariance measures using variables stored in two tensors, with a single
    learnable weight for each time lag.

    The input tensors are interpreted as a set of multivariate observations.
    A covariance estimator computes some measure of statistical dependence
    among the variables in each observation, with the potential addition of
    stochastic noise and dropout to re-weight observations and regularise the
    model.

    Dimension
    ---------
    - Input 1: :math:`(N, *, C_1, O)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      :math:`C_1` denotes number of variables or data channels, O denotes
      number of time points or observations.
    - Input 2: :math:`(N, *, C_2, O)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      :math:`C_2` denotes number of variables or data channels, O denotes
      number of time points or observations.
    - Output: :math:`(N, *, W, C_{*}, C_{*})`
      W denotes number of sets of weights. :math:`C_{*}` can denote either
      :math:`C_1` or :math:`C_2`, depending on the estimator provided. Paired
      estimators produce one axis of each size, while conditional estimators
      produce both axes of size :math:`C_1`.

    Parameters
    ----------
    dim : int
        Number of observations `O` per data instance. This determines the
        dimension of each slice of the covariance weight tensor.
    estimator : callable
        Covariance estimator, e.g. from `hypernova.functional.cov`. The
        estimator must be binary: it should accept two tensors rather than one.
        Some available options are:
        - `pairedcov`: Empirical covariance between variables in tensor 1 and
          those in tensor 2.
        - `pairedcorr`: Pearson correlation between variables in tensor 1 and
          those in tensor 2.
        - `conditionalcov`: Covariance between variables in tensor 1 after
          conditioning on variables in tensor 2. Can be used to control for the
          effects of confounds and is equivalent to confound regression with
          the addition of an intercept term.
        - `conditionalcorr`: Pearson correlation between variables in tensor 1
          after conditioning on variables in tensor 2.
    max_lag : int or None (default 0)
        Maximum lag to include in the weight matrix. If this is not None, the
        structure of the weight matrix is constrained to allow nonzero entries
        only along diagonals that are a maximum offset of `max_lag` from the
        main diagonal. The default value of 0 permits weights only along the
        main diagonal.
    out_channels : int (default 1)
        Number of weight sets to include. For each weight set, the module
        produces an output channel.
    rowvar : bool (default True)
        Indicates that the last axis of the input tensor is the observation
        axis and the penultimate axis is the variable axis. If False, then this
        relationship is transposed.
    bias : bool (default False)
        Indicates that the biased normalisation (i.e., division by `N` in the
        unweighted case) should be performed. By default, normalisation of the
        covariance is unbiased (i.e., division by `N - 1`).
    ddof : int or None (default None)
        Degrees of freedom for normalisation. If this is specified, it
        overrides the normalisation factor automatically determined using the
        `bias` parameter.
    l2: nonnegative float (default 0)
        L2 regularisation term to add to the maximum likelihood estimate of the
        covariance matrix. This can be set to a positive value to obtain an
        intermediate for estimating the regularised inverse covariance or to
        ensure that the covariance matrix is non-singular (if, for instance,
        it needs to be inverted or projected into a tangent space).
    noise: NoiseSource object or None (default None)
        Noise source to inject into the weights. A diagonal noise source adds
        stochasticity into observation weights and can be used to regularise a
        zero-lag covariance. A positive semidefinite noise source ensures that
        the weight tensor remains in the positive semidefinite cone.
    dropout: DropoutSource object or None (default None)
        Dropout source to inject into the weights. A dropout source can be used
        to randomly ignore a subset of observations and thereby perform a type
        of data augmentation (similar to bootstrapped covariance estimates).
    domain: Domain object or None (default None)
        A domain object from `hypernova.functional.domain`, used to specify the
        domain of the weights. An `Identity` object yields the raw weights,
        while objects such as `Logit` domains transform the weights prior to
        multiplication with the input. Using alternative domains can thereby
        constrain the weights to some desirable interval (e.g., [0, +f)).

    Attributes
    ----------
    prepreweight_c, prepreweight_r : Tensor :math:`(W, L)`
        Toeplitz matrix generators for the columns (lag) and rows (lead) of the
        weight matrix. L denotes the maximum lag. These parameters are injected
        along each diagonal of the weight matrix up to the maximum lag. The
        prepreweights are initialised as double exponentials with a maximum of
        1 at the origin (zero lag).
    preweight : Tensor :math:`(W, O, O)`
        Tensor containing raw internal values of the weights. This is the
        preimage of the weights under the transformation specified in the
        `domain` object, prior to the enforcement of any symmetry constraints.
    weight : Tensor :math:`(W, O, O)`
        Tensor containing importance or coupling weights for the observations.
        If this tensor is 1-dimensional, each entry weights the corresponding
        observation in the covariance computation. If it is 2-dimensional,
        then it must be square, symmetric, and positive semidefinite. In this
        case, diagonal entries again correspond to relative importances, while
        off-diagonal entries indicate coupling factors. For instance, a banded
        or multi-diagonal tensor can be used to specify inter-temporal coupling
        for a time series covariance.
    postweight : Tensor :math:`(W, O, O)`
        Final weights as seen by the data. This is the weight after any noise
        and dropout sources are applied and after final masking.
    """
    def __init__(self, dim, estimator, max_lag=0, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, domain=None):
        super(BinaryCovarianceTW, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            domain=domain, out_channels=out_channels
        )


class BinaryCovarianceUW(_BinaryCov, _UnweightedCov):
    """
    Covariance measures using variables stored in two tensors, without
    learnable weights.

    The input tensors are interpreted as a set of multivariate observations.
    A covariance estimator computes some measure of statistical dependence
    among the variables in each observation, with the potential addition of
    stochastic noise and dropout to re-weight observations and regularise the
    model.

    Though it does not contain learnable weights, this module nonetheless
    supports multiple weight channels for the purpose of data augmentation. By
    injecting the weights in different data channels with different noise and
    dropout terms, it can produce several different estimates of covariance
    from the same source data.

    Dimension
    ---------
    - Input 1: :math:`(N, *, C_1, O)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      :math:`C_1` denotes number of variables or data channels, O denotes
      number of time points or observations.
    - Input 2: :math:`(N, *, C_2, O)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      :math:`C_2` denotes number of variables or data channels, O denotes
      number of time points or observations.
    - Output: :math:`(N, *, W, C_{*}, C_{*})`
      W denotes number of sets of weights. :math:`C_{*}` can denote either
      :math:`C_1` or :math:`C_2`, depending on the estimator provided. Paired
      estimators produce one axis of each size, while conditional estimators
      produce both axes of size :math:`C_1`.

    Parameters
    ----------
    dim : int
        Number of observations `O` per data instance. This determines the
        dimension of each slice of the covariance weight tensor.
    estimator : callable
        Covariance estimator, e.g. from `hypernova.functional.cov`. The
        estimator must be binary: it should accept two tensors rather than one.
        Some available options are:
        - `pairedcov`: Empirical covariance between variables in tensor 1 and
          those in tensor 2.
        - `pairedcorr`: Pearson correlation between variables in tensor 1 and
          those in tensor 2.
        - `conditionalcov`: Covariance between variables in tensor 1 after
          conditioning on variables in tensor 2. Can be used to control for the
          effects of confounds and is equivalent to confound regression with
          the addition of an intercept term.
        - `conditionalcorr`: Pearson correlation between variables in tensor 1
          after conditioning on variables in tensor 2.
    out_channels : int (default 1)
        Number of weight sets to include. For each weight set, the module
        produces an output channel.
    rowvar : bool (default True)
        Indicates that the last axis of the input tensor is the observation
        axis and the penultimate axis is the variable axis. If False, then this
        relationship is transposed.
    bias : bool (default False)
        Indicates that the biased normalisation (i.e., division by `N` in the
        unweighted case) should be performed. By default, normalisation of the
        covariance is unbiased (i.e., division by `N - 1`).
    ddof : int or None (default None)
        Degrees of freedom for normalisation. If this is specified, it
        overrides the normalisation factor automatically determined using the
        `bias` parameter.
    l2: nonnegative float (default 0)
        L2 regularisation term to add to the maximum likelihood estimate of the
        covariance matrix. This can be set to a positive value to obtain an
        intermediate for estimating the regularised inverse covariance or to
        ensure that the covariance matrix is non-singular (if, for instance,
        it needs to be inverted or projected into a tangent space).
    noise: NoiseSource object or None (default None)
        Noise source to inject into the weights. A diagonal noise source adds
        stochasticity into observation weights and can be used to regularise a
        zero-lag covariance. A positive semidefinite noise source ensures that
        the weight tensor remains in the positive semidefinite cone.
    dropout: DropoutSource object or None (default None)
        Dropout source to inject into the weights. A dropout source can be used
        to randomly ignore a subset of observations and thereby perform a type
        of data augmentation (similar to bootstrapped covariance estimates).

    Attributes
    ----------
    weight : Tensor :math:`(W, O, O)`
        Tensor containing importance or coupling weights for the observations.
        If this tensor is 1-dimensional, each entry weights the corresponding
        observation in the covariance computation. If it is 2-dimensional,
        then it must be square, symmetric, and positive semidefinite. In this
        case, diagonal entries again correspond to relative importances, while
        off-diagonal entries indicate coupling factors. For instance, a banded
        or multi-diagonal tensor can be used to specify inter-temporal coupling
        for a time series covariance.
    postweight : Tensor :math:`(W, O, O)`
        Final weights as seen by the data. This is the weight after any noise
        and dropout sources are applied and after final masking.
    """
    def __init__(self, dim, estimator, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None):
        super(BinaryCovarianceUW, self).__init__(
            dim=dim, estimator=estimator, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            out_channels=out_channels
        )
