# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Modules supporting covariance estimation.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Literal, Optional
from ..engine import Tensor
from ..engine.paramutil import _to_jax_array
from ..functional import expand_outer, toeplitz


def cfg_banded_parameter(
    max_lag: Optional[int] = 0,
    min_lag: Optional[int] = 0,
    mode: Literal['weight', 'mask'] = 'weight',
    out_channels: int = 1,
):
    mode_fn = {
        'weight': lambda x: jnp.exp(-jnp.arange(x)),
        'mask': lambda x: jnp.ones((x,), dtype=jnp.bool_),
    }
    mode_null = {
        'weight': 0.,
        'mask': False,
    }
    mode_fn = mode_fn[mode]
    null = mode_null[mode]
    if min_lag > max_lag:
        raise ValueError(
            f"min_lag ({min_lag}) must be less than or equal to "
            f"max_lag ({max_lag})."
        )
    if max_lag >= 0 and min_lag <= 0:
        param_row = mode_fn(max_lag + 1)
        param_col = mode_fn(-min_lag + 1)
    elif min_lag > 0:
        param_row = mode_fn(max_lag + 1)
        param_row = param_row.at[:min_lag].set(null)
        param_col = param_row[0]
    elif max_lag < 0:
        param_col = mode_fn(-min_lag + 1)
        param_col = param_col.at[:-max_lag].set(null)
        param_row = param_col[0]
    param_row = jnp.tile(
        param_row[None, ...], (out_channels, 1)
    )
    param_col = jnp.tile(
        param_col[None, ...], (out_channels, 1)
    )
    return param_row, param_col


def document_cov_nn(cls):
    """
    Decorator to document a covariance estimation module.

    Parameters
    ----------
    cls : type
        The class to document.

    Returns
    -------
    type
        The decorated class.
    """
    unary_nn_spec = """
    The input tensor is interpreted as a set of multivariate observations.
    A covariance estimator computes some measure of statistical dependence
    among the variables in each observation, with the potential addition of
    stochastic noise and dropout to re-weight observations and regularise the
    model.

    :Dimension: **Input :** :math:`(N, *, C, O)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, C denotes number of data channels
                    or variables, O denotes number of time points or
                    observations per channel
                **Output :** :math:`(N, *, W, C, C)`
                    W denotes number of sets of weights.
    """

    binary_nn_spec = """
    The input tensors are interpreted as sets of multivariate observations.
    A covariance estimator computes some measure of statistical dependence
    among the variables in each observation, with the potential addition of
    stochastic noise and dropout to re-weight observations and regularise the
    model.

    :Dimension: **Input 1:** :math:`(N, *, C_1, O)`
                    `N` denotes batch size, ``*`` denotes any number of
                    intervening dimensions, :math:`C_1` denotes number of data
                    channels or variables, `O` denotes number of time points
                    or observations per channel.
                **Input 2:** :math:`(N, *, C_2, O)`
                    :math:`C_2` denotes number of data channels or variables
                    in the second input tensor.
                **Output :** :math:`(N, *, W, C_{*}, C_{*})`
                    `W` denotes number of sets of weights. :math:`C_{*}` can
                    denote either :math:`C_1` or :math:`C_2`, depending on the
                    estimator provided. Paired estimators produce one axis of
                    each size, while conditional estimators produce both axes
                    of size :math:`C_1`."""

    weighted_nn_spec = """
    By default, the weight is initialised following a double exponential
    function of lag, such that the weights at 0 lag are
    :math:`e^{{-|0|}} = 1`, the weights at 1 or -1 lag are
    :math:`e^{{-|1|}}`, etc. Note that if the maximum lag is 0, this
    default initialisation will be equivalent to an unweighted covariance."""

    unary_estimator_spec = """
    estimator : callable
        Covariance estimator, e.g. from
        :doc:`hypercoil.functional.cov <hypercoil.functional.cov>`. The
        estimator must be unary: it should accept a single tensor rather than
        multiple tensors. Some available options are:

        - :doc:`cov <hypercoil.functional.cov.cov>`: Raw empirical covariance.
        - :doc:`corr  <hypercoil.functional.cov.corr>`: Pearson correlation.
        - :doc:`precision  <hypercoil.functional.cov.precision>`: Precision.
        - :doc:`partialcorr  <hypercoil.functional.cov.partialcorr>`:
          Partial correlation."""

    binary_estimator_spec = """
    estimator : callable
        Covariance estimator, e.g. from
        :doc:`hypercoil.functional.cov <hypercoil.functional.cov>`. The
        estimator must be binary: it should accept two tensors rather than
        one. Some available options are:

        - :doc:`pairedcov <hypercoil.functional.cov.pairedcov>`:
          Empirical covariance between variables in tensor 1 and those in
          tensor 2.
        - :doc:`pairedcorr <hypercoil.functional.cov.pairedcorr>`:
          Pearson correlation between variables in tensor 1 and those in
          tensor 2.
        - :doc:`conditionalcov <hypercoil.functional.cov.conditionalcov>`:
          Covariance between variables in tensor 1 after conditioning on
          variables in tensor 2. Can be used to control for the effects of
          confounds and is equivalent to confound regression with the
          addition of an intercept term.
        - :doc:`conditionalcorr <hypercoil.functional.cov.conditionalcorr>`:
          Pearson correlation between variables in tensor 1 after
          conditioning on variables in tensor 2."""

    param_spec = """
    dim : int
        Number of observations `O` per data instance. This determines the
        dimension of each slice of the covariance weight tensor.
    min_lag , max_lag : int or None (default 0)
        Minimum and maximum lags to include in the weight matrix. If these
        parameters are not None, the structure of the weight matrix is
        constrained to allow nonzero entries only along diagonals that are a
        maximum offset of (``min_lag``, ``max_lag``) from the main diagonal.
        The default value of 0 permits weights only along the main diagonal.
    out_channels : int (default 1)
        Number of weight sets ``W`` to include. For each weight set, the
        module produces an output channel.
    rowvar : bool (default True)
        Indicates that the last axis of the input tensor is the observation
        axis and the penultimate axis is the variable axis. If False, then
        this relationship is transposed.
    biased : bool (default False)
        Indicates that the biased normalisation (i.e., division by `N` in the
        unweighted case) should be performed. By default, normalisation of the
        covariance is unbiased (i.e., division by `N - 1`).
    ddof : int or None (default None)
        Degrees of freedom for normalisation. If this is specified, it
        overrides the normalisation factor automatically determined using the
        ``biased`` parameter.
    l2 : nonnegative float (default 0)
        L2 regularisation term to add to the maximum likelihood estimate of
        the covariance matrix. This can be set to a positive value to obtain
        intermediate for estimating the regularised inverse covariance or to
        an ensure that the covariance matrix is non-singular (if, for
        instance, it needs to be inverted or projected into a tangent
        space)."""

    weighted_attr_spec = """
    mask : Tensor :math:`(W, O, O)` or None
        Boolean-valued tensor indicating the entries of the weight tensor that
        are permitted to take nonzero values. This is determined by the
        specified ``max_lag`` parameter at initialisation.
    weight : Tensor :math:`(W, O, O)`
        Tensor containing importance or coupling weights for the observations.
        If this tensor is 1-dimensional, each entry weights the corresponding
        observation in the covariance computation. If it is 2-dimensional,
        then it must be square, symmetric, and positive semidefinite. In this
        case, diagonal entries again correspond to relative importances, while
        off-diagonal entries indicate coupling factors. For instance, a banded
        or multi-diagonal tensor can be used to specify inter-temporal
        coupling for a time series covariance."""

    toeplitz_attr_spec = """
    weight_col, weight_row : Tensor :math:`(W, L)`
        Toeplitz matrix generators for the columns (lag) and rows (lead) of
        the weight matrix. L denotes the maximum lag. These parameters are
        repeated along each diagonal of the weight matrix up to the maximum
        lag. The weight generators are initialised as exponentials over
        negative integers with a maximum of 1 at the origin (zero lag;
        :math:`e^0`). The ``weight`` attribute is a property that is generated
        from these parameters as needed."""

    cls.__doc__ = cls.__doc__.format(
        unary_nn_spec=unary_nn_spec,
        binary_nn_spec=binary_nn_spec,
        weighted_nn_spec=weighted_nn_spec,
        unary_estimator_spec=unary_estimator_spec,
        binary_estimator_spec=binary_estimator_spec,
        param_spec=param_spec,
        weighted_attr_spec=weighted_attr_spec,
        toeplitz_attr_spec=toeplitz_attr_spec,
    )
    return cls


class BaseCovariance(eqx.Module):
    """
    Base class for modules that estimate covariance or derived measures.

    ``_Cov`` provides a common initialisation pattern together with methods
    for:

    * injecting noise into the weights to regularise them
    * toggling between train and test modes
    * mapping between the learnable ``'preweight'`` internally stored by the
      module and the weight that is actually ``'seen'`` by the data where this
      is necessary

    Consult specific implementations for comprehensive documentation.
    """
    estimator: Callable
    dim: int
    min_lag: int = 0
    max_lag: int = 0
    out_channels: int = 1
    rowvar: bool = True
    biased: bool = False
    ddof: Optional[int] = None
    l2: float = 0.

    @staticmethod
    def process_parameters(
        weight: Optional[Tensor],
        mask: Optional[Tensor],
    ) -> Tensor:
        if weight is not None:
            weight = _to_jax_array(weight)
        if mask is not None:
            mask = _to_jax_array(mask)
            if weight is None:
                weight = jnp.eye(mask.shape[-1]) # equivalent to unweighted
            if mask.ndim > 1 and mask.shape[-2] == 1:
                mask = mask.swapaxes(-2, -1)
            weight = weight * expand_outer(mask)
        return weight


class UnaryCovMixin:
    """
    Mixin for covariance estimators that operate on a single multivariate
    tensor.

    ``UnaryCovMixin`` provides an implementation of the forward pass through a
    covariance module, which takes as input a single multivariate tensor and
    returns the output of the specified covariance estimator, applied to the
    input tensor.
    """
    def __call__(
        self,
        input: Tensor,
        weight: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        if input.ndim > 2 and self.out_channels > 1 and input.shape[-3] > 1:
            input = input[..., None, :, :]
        weight = self.process_parameters(weight=weight, mask=mask)
        return self.estimator(
            input,
            rowvar=self.rowvar,
            bias=self.biased,
            ddof=self.ddof,
            weight=weight,
            l2=self.l2
        )


class BinaryCovMixin:
    """
    Base class for covariance estimators that operate on a pair of
    multivariate tensors.

    ``BinaryCovMixin`` provides an implementation of the forward pass through
    a covariance module, which takes as input two multivariate tensors and
    returns the output of the specified covariance estimator, applied to the
    input tensor pair.
    """
    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        weight: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        if self.out_channels > 1:
            if x.ndim > 2 and x.shape[-3] > 1:
                x = x[..., None, :, :]
            if y.ndim > 2 and y.shape[-3] > 1:
                y = y[..., None, :, :]
        weight = self.process_parameters(weight=weight, mask=mask)
        return self.estimator(
            x, y,
            rowvar=self.rowvar,
            bias=self.biased,
            ddof=self.ddof,
            weight=weight,
            l2=self.l2
        )


class ParameterisedUnaryCovMixin(UnaryCovMixin):
    """
    Like :class:`UnaryCovMixin`, but with weight and mask parameters that are
    attributes of the module.
    """
    def __call__(
        self,
        input: Tensor,
        mask: Optional[Tensor] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if self.mask is not None and mask is not None:
            mask = self.mask & mask
        elif mask is None:
            mask = self.mask
        return super().__call__(
            input=input,
            weight=self.weight,
            mask=mask,
            key=key,
        )


class ParameterisedBinaryCovMixin(BinaryCovMixin):
    """
    Like :class:`BinaryCovMixin`, but with weight and mask parameters that are
    attributes of the module.
    """
    def __call__(
        self,
        x: Tensor,
        y: Tensor,
        mask: Optional[Tensor] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        if self.mask is not None and mask is not None:
            mask = self.mask & mask
        elif mask is None:
            mask = self.mask
        return super().__call__(
            x=x,
            y=y,
            weight=self.weight,
            mask=mask,
            key=key,
        )


class BaseWeightedCovariance(BaseCovariance):
    """
    Base class for covariance estimators with a full complement of learnable
    weights.

    ``_WeightedCov`` extends ``BaseCovariance`` by providing a default
    initialisation framework for the module's learnable parameters.

    Consult specific implementations for comprehensive documentation.
    """
    weight: Tensor
    mask: Optional[Tensor] = None

    def __init__(
        self,
        estimator: Callable,
        dim: int,
        min_lag: int = 0,
        max_lag: int = 0,
        out_channels: int = 1,
        rowvar: bool = True,
        biased: bool = False,
        ddof: Optional[int] = None,
        l2: float = 0,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        super().__init__(
            dim=dim,
            estimator=estimator,
            min_lag=min_lag,
            max_lag=max_lag,
            out_channels=out_channels,
            rowvar=rowvar,
            biased=biased,
            ddof=ddof,
            l2=l2,
        )
        mask = None
        if min_lag is None: min_lag = -(dim - 1)
        if max_lag is None: max_lag = dim - 1
        if max_lag == 0 and min_lag == 0:
            weight = jnp.ones((out_channels, 1, dim))
        else:
            vals_r, vals_c = cfg_banded_parameter(
                max_lag=max_lag,
                min_lag=min_lag,
                mode='weight',
                out_channels=out_channels,
            )
            weight = toeplitz(
                c=vals_c,
                r=vals_r,
                shape=(dim, dim),
                fill_value=0.,
            )
            if max_lag is not None or min_lag is not None:
                mask_vals_r, mask_vals_c = cfg_banded_parameter(
                    max_lag=max_lag,
                    min_lag=min_lag,
                    mode='mask',
                    out_channels=out_channels,
                )
                mask = toeplitz(
                    c=mask_vals_c,
                    r=mask_vals_r,
                    shape=(dim, dim),
                    fill_value=False,
                )
        self.weight = weight
        self.mask = mask


class BaseToeplitzWeightedCovariance(BaseCovariance):
    """
    Base class for covariance estimators with a single learnable weight for
    each time lag.
    """
    weight_col: Tensor
    weight_row: Tensor
    mask: Optional[Tensor] = None

    #TODO: Replace this entire thing with a convolution-based implementation.
    # That should make it much, much faster but will require a separate
    # forward pass...
    def __init__(
        self,
        estimator: Callable,
        dim: int,
        min_lag: int = 0,
        max_lag: int = 0,
        out_channels: int = 1,
        rowvar: bool = True,
        biased: bool = False,
        ddof: Optional[int] = None,
        l2: float = 0.,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        super().__init__(
            dim=dim,
            estimator=estimator,
            min_lag=0,
            max_lag=max_lag,
            out_channels=out_channels,
            rowvar=rowvar,
            biased=biased,
            ddof=ddof,
            l2=l2,
        )
        mask = None
        if min_lag is None: min_lag = -(dim - 1)
        if max_lag is None: max_lag = dim - 1
        if max_lag is not None or min_lag is not None:
            mask_row, mask_col = cfg_banded_parameter(
                max_lag=max_lag,
                min_lag=min_lag,
                mode='mask',
                out_channels=out_channels,
            )
            mask = toeplitz(
                c=mask_col,
                r=mask_row,
                shape=(dim, dim),
                fill_value=False,
            )
        weight_row, weight_col = cfg_banded_parameter(
            max_lag=max_lag,
            min_lag=min_lag,
            mode='weight',
            out_channels=out_channels,
        )
        self.weight_col = weight_col
        self.weight_row = weight_row
        self.mask = mask

    @property
    def weight(self):
        return toeplitz(
            c=_to_jax_array(self.weight_col),
            r=_to_jax_array(self.weight_row),
            shape=(self.dim, self.dim),
            fill_value=0.,
        )


#TODO: I hate having these __init__ methods -- feels very much like an anti-
# pattern. But I can't think of a better way to do it right now.


@document_cov_nn
class UnaryCovariance(
    BaseWeightedCovariance,
    ParameterisedUnaryCovMixin,
):
    """
    Covariance measures of a single tensor, with a full complement of
    learnable weights.
    \
    {weighted_nn_spec}\
    {unary_nn_spec}

    Parameters
    ----------\
    {unary_estimator_spec}\
    {param_spec}

    Attributes
    ----------\
    {weighted_attr_spec}
    """
    def __init__(
        self,
        estimator: Callable,
        dim: int,
        min_lag: int = 0,
        max_lag: int = 0,
        out_channels: int = 1,
        rowvar: bool = True,
        biased: bool = False,
        ddof: Optional[int] = None,
        l2: float = 0,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        super().__init__(
            estimator=estimator,
            dim=dim,
            min_lag=min_lag,
            max_lag=max_lag,
            out_channels=out_channels,
            rowvar=rowvar,
            biased=biased,
            ddof=ddof,
            l2=l2,
            key=key,
        )


@document_cov_nn
class UnaryCovarianceTW(
    BaseToeplitzWeightedCovariance,
    ParameterisedUnaryCovMixin,
):
    """
    Covariance measures of a single tensor, with a single learnable weight for
    each time lag.
    \
    {weighted_nn_spec}\
    {unary_nn_spec}

    Parameters
    ----------\
    {unary_estimator_spec}\
    {param_spec}

    Attributes
    ----------\
    {toeplitz_attr_spec}\
    {weighted_attr_spec}
    """
    def __init__(
        self,
        estimator: Callable,
        dim: int,
        min_lag: int = 0,
        max_lag: int = 0,
        out_channels: int = 1,
        rowvar: bool = True,
        biased: bool = False,
        ddof: Optional[int] = None,
        l2: float = 0,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        super().__init__(
            estimator=estimator,
            dim=dim,
            min_lag=min_lag,
            max_lag=max_lag,
            out_channels=out_channels,
            rowvar=rowvar,
            biased=biased,
            ddof=ddof,
            l2=l2,
            key=key,
        )


@document_cov_nn
class UnaryCovarianceUW(
    BaseCovariance,
    UnaryCovMixin,
):
    """
    Covariance measures of a single tensor, without learnable weights.
    \
    {unary_nn_spec}

    Parameters
    ----------\
    {unary_estimator_spec}\
    {param_spec}
    """
    def __init__(
        self,
        estimator: Callable,
        dim: int,
        min_lag: int = 0,
        max_lag: int = 0,
        out_channels: int = 1,
        rowvar: bool = True,
        biased: bool = False,
        ddof: Optional[int] = None,
        l2: float = 0,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        super().__init__(
            estimator=estimator,
            dim=dim,
            min_lag=min_lag,
            max_lag=max_lag,
            out_channels=out_channels,
            rowvar=rowvar,
            biased=biased,
            ddof=ddof,
            l2=l2,
        )


@document_cov_nn
class BinaryCovariance(
    BaseWeightedCovariance,
    ParameterisedBinaryCovMixin,
):
    r"""
    Covariance measures using variables stored in two tensors, with a full
    complement of learnable weights.
    \
    {weighted_nn_spec}\
    {binary_nn_spec}

    Parameters
    ----------\
    {binary_estimator_spec}\
    {param_spec}

    Attributes
    ----------\
    {weighted_attr_spec}
    """
    def __init__(
        self,
        estimator: Callable,
        dim: int,
        min_lag: int = 0,
        max_lag: int = 0,
        out_channels: int = 1,
        rowvar: bool = True,
        biased: bool = False,
        ddof: Optional[int] = None,
        l2: float = 0,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        super().__init__(
            estimator=estimator,
            dim=dim,
            min_lag=min_lag,
            max_lag=max_lag,
            out_channels=out_channels,
            rowvar=rowvar,
            biased=biased,
            ddof=ddof,
            l2=l2,
            key=key,
        )


@document_cov_nn
class BinaryCovarianceTW(
    BaseToeplitzWeightedCovariance,
    ParameterisedBinaryCovMixin,
):
    r"""
    Covariance measures using variables stored in two tensors, with a single
    learnable weight for each time lag.
    \
    {weighted_nn_spec}\
    {binary_nn_spec}

    Parameters
    ----------\
    {binary_estimator_spec}\
    {param_spec}

    Attributes
    ----------\
    {toeplitz_attr_spec}\
    {weighted_attr_spec}
    """
    def __init__(
        self,
        estimator: Callable,
        dim: int,
        min_lag: int = 0,
        max_lag: int = 0,
        out_channels: int = 1,
        rowvar: bool = True,
        biased: bool = False,
        ddof: Optional[int] = None,
        l2: float = 0,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        super().__init__(
            estimator=estimator,
            dim=dim,
            min_lag=min_lag,
            max_lag=max_lag,
            out_channels=out_channels,
            rowvar=rowvar,
            biased=biased,
            ddof=ddof,
            l2=l2,
            key=key,
        )


@document_cov_nn
class BinaryCovarianceUW(
    BaseCovariance,
    BinaryCovMixin,
):
    r"""
    Covariance measures using variables stored in two tensors, without
    learnable weights.
    \
    {binary_nn_spec}

    Parameters
    ----------\
    {binary_estimator_spec}\
    {param_spec}
    """
    def __init__(
        self,
        estimator: Callable,
        dim: int,
        min_lag: int = 0,
        max_lag: int = 0,
        out_channels: int = 1,
        rowvar: bool = True,
        biased: bool = False,
        ddof: Optional[int] = None,
        l2: float = 0,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        super().__init__(
            estimator=estimator,
            dim=dim,
            min_lag=min_lag,
            max_lag=max_lag,
            out_channels=out_channels,
            rowvar=rowvar,
            biased=biased,
            ddof=ddof,
            l2=l2,
        )
