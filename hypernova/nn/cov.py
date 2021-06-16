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
from ..init.base import (
    BaseInitialiser,
    ConstantInitialiser,
    identity_init_,
)
from ..init.laplace import LaplaceInit
from ..init.toeplitz import ToeplitzInit


class _Cov(Module):
    """
    Base class for modules that estimate covariance or derived measures.

    `_Cov` provides a common initialisation pattern together with methods for:
    * injecting noise into the weights to regularise them
    * toggling between train and test modes
    * mapping between the learnable 'preweight' internally stored by the
      module and the weight that is actually 'seen' by the data where this
      is necessary

    Consult specific implementations for comprehensive documentation.
    """
    def __init__(self, dim, estimator, max_lag=0, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None):
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

        if self.max_lag is None or self.max_lag == 0:
            self.register_parameter('mask', None)

    def inject_noise(self, weight):
        """
        Inject noise into a weight tensor, as determined by the module's
        `noise` and `dropout` attributes.

        Noise injection can be used to regularise the module or to augment
        the dataset by jittering covariance estimates.

        The `noise` attribute controls the distribution of additive noise,
        while the `dropout` attribute controls the distribution of
        multiplicative noise.
        """
        if self.noise is not None:
            weight = self.noise.inject(weight)
        if self.dropout is not None:
            weight = self.dropout.inject(weight)
        return weight

    def train(self, mode=True):
        """
        Toggle the module between training and testing modes. If `mode` is set
        to `False`, then the module enters testing mode; if it is `True` or
        not explicitly specified, then the module enters training mode.
        """
        super(_Cov, self).train(mode)
        if self.noise is not None:
            self.noise.train(mode)
        if self.dropout is not None:
            self.dropout.train(mode)

    def eval(self):
        """Switch the module into testing mode."""
        super(_Cov, self).eval()
        if self.noise is not None:
            self.noise.eval()
        if self.dropout is not None:
            self.dropout.eval()

    def extra_repr(self):
        s = f'estimator={self.estimator.__name__}, dim={self.dim}'
        if self.out_channels > 1:
            s += f', channels={self.out_channels}'
        if self.max_lag > 0:
            s += f', max_lag={self.max_lag}'
        if not self.rowvar:
            s += f', format=column'
        if self.bias:
            s += f', biased estimator'
        if self.ddof is not None:
            s += f', ddof={self.ddof}'
        if self.l2 > 0:
            s += f', l2={self.l2}'
        return s

    @property
    def weight(self):
        return self.init.domain.image(self.preweight)

    @property
    def postweight(self):
        return self.inject_noise(self.weight)


class _UnaryCov(_Cov):
    """
    Base class for covariance estimators that operate on a single variable
    tensor.

    `_UnaryCov` extends `_Cov` by providing an implementation of the forward
    pass through the module, which takes as input a variable tensor and
    returns the output of the specified covariance estimator, applied to the
    input tensor.

    Consult specific implementations for comprehensive documentation.
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

    `_BinaryCov` extends `_Cov` by providing an implementation of the forward
    pass through the module, which takes as input two variable tensors and
    returns the output of the specified covariance estimator, applied to the
    input tensor pair.

    Consult specific implementations for comprehensive documentation.
    """
    def forward(self, x, y):
        if self.out_channels > 1:
            if x.dim() > 2 and x.size(-3) > 1:
                x = x.unsqueeze(-3)
            if y.dim() > 2 and y.size(-3) > 1:
                y = y.unsqueeze(-3)
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

    `_WeightedCov` extends `_Cov` by providing a default initialisation
    framework for the module's learnable parameters.

    Consult specific implementations for comprehensive documentation.
    """
    def __init__(self, dim, estimator, max_lag=0, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, init=None):
        super(_WeightedCov, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            out_channels=out_channels
        )
        if self.max_lag == 0:
            self.init = init or ConstantInitialiser(1, domain=Identity())
            self.preweight = Parameter(torch.Tensor(
                self.out_channels, 1, self.dim
            ))
        else:
            vals = laplace(torch.arange(self.max_lag + 1))
            self.init = init or ToeplitzInit(
                c=vals,
                fill_value=0,
                domain=Identity()
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
        self.init(self.preweight)
        if self.max_lag is not None and self.max_lag != 0:
            mask_vals = torch.Tensor([1 for _ in range(self.max_lag + 1)])
            mask_init = ToeplitzInit(c=mask_vals, fill_value=0)
            mask_init(self.mask)

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
    #TODO: Replace this entire thing with a convolution-based implementation.
    # That should make it much, much faster but will require a separate
    # forward pass...
    def __init__(self, dim, estimator, max_lag=0, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, init=None):
        super(_ToeplitzWeightedCov, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            out_channels=out_channels
        )
        if self.max_lag is not None:
            self.mask = Parameter(torch.Tensor(
                self.dim, self.dim
            ).bool(), requires_grad=False)
        self.init = init or LaplaceInit(
            loc=(0, 0), excl_axis=[1], domain=Identity()
        )
        self.prepreweight_c = Parameter(torch.Tensor(
            self.max_lag + 1, self.out_channels
        ))
        self.prepreweight_r = Parameter(torch.Tensor(
            self.max_lag + 1, self.out_channels
        ))
        self.reset_parameters()

    def reset_parameters(self):
        if self.max_lag is not None:
            mask_vals = torch.Tensor([1 for _ in range(self.max_lag + 1)])
            mask_init = ToeplitzInit(c=mask_vals, fill_value=0)
        self.init(self.prepreweight_c)
        self.init(self.prepreweight_r)
        mask_init(self.mask)

    @property
    def preweight(self):
        return toeplitz(
            c=self.prepreweight_c,
            r=self.prepreweight_r,
            dim=(self.dim, self.dim),
            fill_value=self.init.domain.preimage(torch.tensor(0.)).item()
        )

    @property
    def postweight(self):
        if self.mask is not None:
            return self.inject_noise(self.weight) * self.mask
        return self.inject_noise(self.weight)


class _UnweightedCov(_Cov):
    """
    Base class for covariance estimators without learnable parameters.

    `_UnweightedCov` extends `_Cov` by initialising all weights to identity
    (equivalent to unweighted covariance).

    Consult specific implementations for comprehensive documentation.
    """
    def __init__(self, dim, estimator, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None):
        super(_UnweightedCov, self).__init__(
            dim=dim, estimator=estimator, max_lag=0, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            out_channels=out_channels
        )
        self.init = BaseInitialiser(init=identity_init_)
        self.preweight = Parameter(
            torch.Tensor(self.out_channels, self.dim, self.dim),
            requires_grad=False
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.init(self.preweight)


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
        Number of weight sets `W` to include. For each weight set, the module
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
    init: Initialiser object or None (default None)
        An initialiser object from `hypernova.init`, used to specify the
        initialisation scheme for the weights. If none is otherwise provided,
        this defaults to initialising weights following a double exponential
        function of lag, such that the weights at 0 lag are e^-|0| = 1, the
        weights at 1 or -1 lag are e^-|1|, etc. Note that if the maximum lag
        is 0, this default initialisation will be equivalent to an unweighted
        covariance.

    Attributes
    ----------
    mask : Tensor :math:`(W, O, O)`
        Boolean-valued tensor indicating the entries of the weight tensor that
        are permitted to take nonzero values. This is determined by the
        specified `max_lag` parameter at initialisation.
    preweight : Tensor :math:`(W, O, O)`
        Tensor containing raw internal values of the weights. This is the
        preimage of the weights under any transformation specified in the
        `init` object, prior to the enforcement of any symmetry constraints.
        By default, the preweight is thus initialised as the preimage of a
        Toeplitz banded matrix where the weight of each diagonal is set
        according to a double exponential function with a maximum of 1 at the
        origin (zero lag).
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
        and dropout sources are injected and after final masking.
    """
    def __init__(self, dim, estimator, max_lag=0, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, init=None):
        super(UnaryCovariance, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            init=init, out_channels=out_channels
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
        Number of weight sets `W` to include. For each weight set, the module
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
    init: Initialiser object or None (default None)
        An initialiser object from `hypernova.init`, used to specify the
        initialisation scheme for the weights. If none is otherwise provided,
        this defaults to initialising weights following a double exponential
        function of lag, such that the weights at 0 lag are e^-|0| = 1, the
        weights at 1 or -1 lag are e^-|1|, etc. Note that if the maximum lag
        is 0, this default initialisation will be equivalent to an unweighted
        covariance.

    Attributes
    ----------
    prepreweight_c, prepreweight_r : Tensor :math:`(W, L)`
        Toeplitz matrix generators for the columns (lag) and rows (lead) of the
        weight matrix. L denotes the maximum lag. These parameters are repeated
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
        and dropout sources are injected and after final masking.
    """
    def __init__(self, dim, estimator, max_lag=0, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, init=None):
        super(UnaryCovarianceTW, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            init=init, out_channels=out_channels
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
        Number of weight sets `W` to include. For each weight set, the module
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
        and dropout sources are injected and after final masking.
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
        Number of weight sets `W` to include. For each weight set, the module
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
    init: Initialiser object or None (default None)
        An initialiser object from `hypernova.init`, used to specify the
        initialisation scheme for the weights. If none is otherwise provided,
        this defaults to initialising weights following a double exponential
        function of lag, such that the weights at 0 lag are e^-|0| = 1, the
        weights at 1 or -1 lag are e^-|1|, etc. Note that if the maximum lag
        is 0, this default initialisation will be equivalent to an unweighted
        covariance.

    Attributes
    ----------
    mask : Tensor :math:`(W, O, O)`
        Boolean-valued tensor indicating the entries of the weight tensor that
        are permitted to take nonzero values. This is determined by the
        specified `max_lag` parameter at initialisation.
    preweight : Tensor :math:`(W, O, O)`
        Tensor containing raw internal values of the weights. This is the
        preimage of the weights under any transformation specified in the
        `init` object, prior to the enforcement of any symmetry constraints.
        By default, the preweight is thus initialised as the preimage of a
        Toeplitz banded matrix where the weight of each diagonal is set
        according to a double exponential function with a maximum of 1 at the
        origin (zero lag).
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
        and dropout sources are injected and after final masking.
    """
    def __init__(self, dim, estimator, max_lag=0, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, init=None):
        super(BinaryCovariance, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            init=init, out_channels=out_channels
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
        Number of weight sets `W` to include. For each weight set, the module
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
    init: Initialiser object or None (default None)
        An initialiser object from `hypernova.init`, used to specify the
        initialisation scheme for the weights. If none is otherwise provided,
        this defaults to initialising weights following a double exponential
        function of lag, such that the weights at 0 lag are e^-|0| = 1, the
        weights at 1 or -1 lag are e^-|1|, etc. Note that if the maximum lag
        is 0, this default initialisation will be equivalent to an unweighted
        covariance.

    Attributes
    ----------
    prepreweight_c, prepreweight_r : Tensor :math:`(W, L)`
        Toeplitz matrix generators for the columns (lag) and rows (lead) of the
        weight matrix. L denotes the maximum lag. These parameters are repeated
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
        and dropout sources are injected and after final masking.
    """
    def __init__(self, dim, estimator, max_lag=0, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None, init=None):
        super(BinaryCovarianceTW, self).__init__(
            dim=dim, estimator=estimator, max_lag=max_lag, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            init=init, out_channels=out_channels
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
        Number of weight sets `W` to include. For each weight set, the module
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
        and dropout sources are injected and after final masking.
    """
    def __init__(self, dim, estimator, out_channels=1,
                 rowvar=True, bias=False, ddof=None, l2=0,
                 noise=None, dropout=None):
        super(BinaryCovarianceUW, self).__init__(
            dim=dim, estimator=estimator, rowvar=rowvar,
            bias=bias, ddof=ddof, l2=l2, noise=noise, dropout=dropout,
            out_channels=out_channels
        )
