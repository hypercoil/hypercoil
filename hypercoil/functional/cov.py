# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Differentiable estimation of covariance and derived measures.

Functional connectivity is a measure of the statistical relationship between
(localised) time series signals. It is typically operationalised as some
derivative of the covariance between the time series, most often the Pearson
correlation.
"""
import jax
import jax.numpy as jnp
from typing import Literal, Optional, Sequence, Tuple, Union
from .utils import Tensor, _conform_vector_weight, vmap_over_outer


def document_covariance(func):
    param_spec = """
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
    weight : Tensor or None (default None)
        Tensor containing importance or coupling weights for the observations.
        If this tensor is 1-dimensional, each entry weights the corresponding
        observation in the covariance computation. If it is 2-dimensional,
        then it must be square, symmetric, and positive semidefinite. In this
        case, diagonal entries again correspond to relative importances, while
        off-diagonal entries indicate coupling factors. For instance, a banded
        or multi-diagonal tensor can be used to specify inter-temporal
        coupling for a time series covariance.
    l2 : nonnegative float (default 0)
        L2 regularisation term to add to the maximum likelihood estimate of
        the covariance matrix. This can be set to a positive value to obtain
        an intermediate for estimating the regularised inverse covariance."""
    unary_param_spec = """
    X : Tensor
        Tensor containing a sample of multivariate observations. Each slice
        along the last axis corresponds to an observation, and each slice along
        the penultimate axis corresponds to a data channel or more generally a
        variable."""
    binary_param_spec = """
    X, Y : Tensors
        Tensors containing samples of multivariate observations. Each slice
        along the last axis corresponds to an observation, and each slice along
        the penultimate axis corresponds to a data channel or more generally a
        variable."""
    conditional_param_spec = """
    X : Tensor
        Tensor containing samples of multivariate observations, with those
        variables whose influence we wish to control for removed and separated
        out into tensor Y. Each slice along the last axis corresponds to an
        observation, and each slice along the penultimate axis corresponds to a
        data channel or more generally a variable.
    Y : Tensor
        Tensor containing samples of multivariate observations, limited to
        nuisance or confound variables whose influence we wish to control for.
        Each slice along the last axis corresponds to an observation, and each
        slice along the penultimate axis corresponds to a data channel or more
        generally a variable."""
    inverting_param_spec = """
    require_nonsingular : bool
        Indicates that the covariance must be nonsingular. If this is False,
        then the Moore-Penrose pseudoinverse is computed instead of the
        inverse."""
    unary_dim_spec = """**Input :** :math:`(N, *, C, obs)` or :math:`(N, *, obs, C)`
                    N denotes batch size, `*` denotes any number of
                    intervening dimensions, C denotes number of data channels
                    or variables to be correlated, obs denotes number of
                    observations per channel
                **Weight :** :math:`(obs)` or :math:`(obs, obs)`
                    As above
                **Output :** :math:`(N, *, C, C)`
                    As above"""
    binary_dim_spec = """**Input X:** :math:`(N, *, C_X, obs)` or :math:`(N, *, obs, C_X)`
                    N denotes batch size, `*` denotes any number of
                    intervening dimensions, :math:`C_X` denotes number of data
                    channels or variables to be correlated, obs denotes number
                    of observations per channel
                **Input Y:** :math:`(N, *, C_Y, obs)` or :math:`(N, *, obs, C_Y)`
                    :math:`C_Y` denotes number of data channels or variables
                    to be correlated (or conditioned on)
                **Weight :** :math:`(obs)` or :math:`(obs, obs)`
                    As above
                **Output :** :math:`(N, *, C_X, C_Y)`
                    As above"""
    func.__doc__ = func.__doc__.format(
        param_spec=param_spec,
        unary_param_spec=unary_param_spec,
        binary_param_spec=binary_param_spec,
        conditional_param_spec=conditional_param_spec,
        inverting_param_spec=inverting_param_spec,
        unary_dim_spec=unary_dim_spec,
        binary_dim_spec=binary_dim_spec
    )
    return func


@document_covariance
def cov(
    X: Tensor,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    weight: Optional[Tensor] = None,
    l2: float = 0
) -> Tensor:
    """
    Empirical covariance of variables in a tensor batch.

    :Dimension: {unary_dim_spec}

    Parameters
    ----------\
    {unary_param_spec}\
    {param_spec}

    Returns
    -------
    sigma : Tensor
        Empirical covariance matrix of the variables in the input tensor.

    See also
    --------
    pairedcov : Covariance among variables in 2 tensors
    corr: Normalised covariance matrix (Pearson correlation matrix)
    precision: Inverse covariance (precision) matrix
    partialcorr: Partial correlation matrix
    """
    X = _prepare_input(X, rowvar)
    weight, w_type, w_sum, (avg,) = _prepare_weight_and_avg((X,), weight)
    fact = _prepare_denomfact(w_sum, w_type, ddof, bias, weight)

    X0 = X - avg
    if weight is None:
        sigma = X0 @ X0.swapaxes(-1, -2) / fact
    elif w_type == 'vector':
        sigma = (X0 * weight) @ X0.swapaxes(-1, -2) / fact
    else:
        sigma = X0 @ weight @ X0.swapaxes(-1, -2) / fact
    if l2 > 0:
        sigma = sigma + l2 * jnp.eye(X.shape[-2])
    return sigma


@document_covariance
def corr(
    X: Tensor,
    **params
) -> Tensor:
    r"""
    Pearson correlation of variables in a tensor batch.

    The correlation is obtained via normalisation of the covariance. Given a
    covariance matrix
    :math:`\hat{{\Sigma}} \in \mathbb{{R}}^{{n \times n}}`, each
    entry of the correlation matrix
    :math:`R \in \mathbb{{R}}^{{n \times n}}`
    is defined according to

    :math:`R_{{ij}} = \frac{{\hat{{\Sigma}}_{{ij}}}}{{\sqrt{{\hat{{\Sigma}}_{{ii}}}} \sqrt{{\hat{{\Sigma}}_{{jj}}}}`

    :Dimension: {unary_dim_spec}

    Parameters
    ----------\
    {unary_param_spec}\
    {param_spec}

    Returns
    -------
    R : Tensor
        Pearson correlation matrix of the variables in the input tensor.

    See also
    --------
    cov: Empirical covariance matrix
    partialcorr: Partial correlation matrix
    conditionalcorr: Conditional correlation matrix
    """
    sigma = cov(X, **params)
    fact = corrnorm(sigma)
    return sigma / fact


@document_covariance
def partialcov(
    X: Tensor,
    require_nonsingular: bool = True,
    **params
) -> Tensor:
    """
    Partial covariance of variables in a tensor batch.

    The partial covariance is obtained by conditioning the covariance of each
    pair of variables on all other observed variables. It can be interpreted
    as a measurement of the direct relationship between each variable pair.
    The partial covariance is computed via inversion of the covariance matrix,
    followed by negation of off-diagonal entries.

    :Dimension: {unary_dim_spec}

    Parameters
    ----------\
    {unary_param_spec}\
    {inverting_param_spec}\
    {param_spec}

    Returns
    -------
    sigma : Tensor
        Partial covariance matrix of the variables in the input tensor.

    See also
    --------
    cov: Empirical covariance matrix
    partialcorr: Partial correlation matrix
    conditionalcov: Conditional covariance matrix
    precision: Inverse covariance (precision) matrix
    """
    omega = precision(X, require_nonsingular=require_nonsingular, **params)
    omega = omega * (
        2 * jnp.eye(omega.shape[-1]) - 1
    )
    return omega


@document_covariance
def partialcorr(
    X: Tensor,
    require_nonsingular: bool = True,
    **params
) -> Tensor:
    """
    Partial Pearson correlation of variables in a tensor batch.

    The partial correlation is obtained by conditioning the covariance of each
    pair of variables on all other observed variables. It can be interpreted
    as a measurement of the direct relationship between each variable pair.
    The partial correlation is efficiently computed via successive inversion
    and normalisation of the covariance matrix, accompanied by negation of
    off-diagonal entries.

    :Dimension: {unary_dim_spec}

    Parameters
    ----------
    ----------\
    {unary_param_spec}\
    {inverting_param_spec}\
    {param_spec}

    Returns
    -------
    R : Tensor
        Partial Pearson correlation matrix of the variables in the input tensor.

    See also
    --------
    corr: Pearson correlation matrix
    partialcov: Partial covariance matrix
    conditionalcorr: Conditional correlation matrix
    """
    omega = partialcov(X, require_nonsingular=require_nonsingular, **params)
    fact = corrnorm(omega)
    return omega / fact


@document_covariance
def pairedcov(
    X: Tensor,
    Y: Tensor,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    weight: Optional[Tensor] = None,
    l2: float = 0
) -> Tensor:
    """
    Empirical covariance between two sets of variables.

    .. danger::
        The ``l2`` parameter has no effect on this function. It is included only
        for conformance with the ``cov`` function.

    :Dimension: {binary_dim_spec}

    Parameters
    ----------\
    {binary_param_spec}\
    {param_spec}

    Returns
    -------
    sigma : Tensor
        Covariance matrix of the variables in the input tensor.

    See also
    --------
    cov: Empirical covariance matrix
    pairedcorr: Paired Pearson correlation matrix
    conditionalcov: Conditional covariance matrix
    """
    X = _prepare_input(X, rowvar)
    Y = _prepare_input(Y, rowvar)
    weight, w_type, w_sum, (Xavg, Yavg) = _prepare_weight_and_avg((X, Y), weight)
    fact = _prepare_denomfact(w_sum, w_type, ddof, bias, weight)

    X0 = X - Xavg
    Y0 = Y - Yavg
    if weight is None:
        sigma = X0 @ Y0.swapaxes(-1, -2) / fact
    elif w_type == 'vector':
        sigma = (X0 * weight) @ Y0.swapaxes(-1, -2) / fact
    else:
        sigma = X0 @ weight @ Y0.swapaxes(-1, -2) / fact
    return sigma


@document_covariance
def pairedcorr(
    X: Tensor,
    Y: Tensor,
    rowvar: bool = True,
    bias: bool = False,
    ddof: Optional[int] = None,
    **params
) -> Tensor:
    """
    Empirical Pearson correlation between variables in two tensor batches.

    The empirical paired correlation is obtained via normalisation of the
    paired covariance. Given a paired covariance matrix
    :math:`\hat{{\Sigma}} \in \mathbb{{R}}^{{n \\times m}}`, each
    entry of the paired correlation matrix
    :math:`R \in \mathbb{{R}}^{{n \\times m}}`
    is defined according to

    :math:`R_{{ij}} = \\frac{{\hat{{\Sigma}}_{{ij}}}}{{\sqrt{{\hat{{\Sigma}}_{{ii}}}} \sqrt{{\hat{{\Sigma}}_{{jj}}}}`

    .. danger::
        The ``l2`` parameter has no effect on this function. It is included only
        for conformance with the ``cov`` function.

    :Dimension: {binary_dim_spec}

    Parameters
    ----------\
    {binary_param_spec}\
    {param_spec}

    Returns
    -------
    R : Tensor
        Paired Pearson correlation matrix of the variables in the input tensor.
    """
    inddof = ddof
    if inddof is None:
        inddof = 1 - bias
    varX = X.var(-1, keepdims=True, ddof=inddof)
    varY = Y.var(-1, keepdims=True, ddof=inddof)
    fact = jax.lax.sqrt(varX @ varY.swapaxes(-2, -1))
    return pairedcov(X, Y, rowvar=rowvar, bias=bias, ddof=ddof, **params) / fact


def conditionalcov(
    X: Tensor,
    Y: Tensor,
    require_nonsingular: bool = True,
    **params
) -> Tensor:
    """
    Conditional covariance of variables in a tensor batch.

    The conditional covariance is the covariance of one set of variables X
    conditioned on another set of variables Y. The conditional covariance is
    computed from the covariance :math:`\Sigma_{{XX}}` of X ,
    the covariance :math:`\Sigma_{{YY}}` of Y, and the
    covariance :math:`\Sigma_{{XY}}` between X and Y.
    It is defined as the Schur complement of :math:`\Sigma_{{YY}}`:

    :math:`\Sigma_{{X|Y}} = \Sigma_{{XX}} - \Sigma_{{XY}} \Sigma_{{YY}}^{{-1}} \Sigma_{{XY}}^\intercal`

    The conditional covariance is equivalent to the covariance of the first set
    of variables after residualising them with respect to the second set of
    variables (plus an intercept term). This can be interpreted as the
    covariance of variables of interest (the first set) after controlling for
    the effects of confounds or nuisance variables (the second set).

    :Dimension: {binary_dim_spec}

    Parameters
    ----------\
    {conditional_param_spec}\
    {inverting_param_spec}\
    {param_spec}

    Returns
    -------
    sigma : Tensor
        Conditional empirical covariance matrix of the variables in input
        tensor X conditioned on the variables in input tensor Y.

    See also
    --------
    conditionalcorr: Normalised conditional covariance (Pearson correlation)
    partialcov: Condition each variable on all other variables
    """
    A = cov(X, **params)
    B = pairedcov(X, Y, **params)
    C_inv = precision(Y, require_nonsingular=require_nonsingular, **params)
    return A - B @ C_inv @ B.swapaxes(-1, -2)


@document_covariance
def conditionalcorr(
    X: Tensor,
    Y: Tensor,
    require_nonsingular: bool = True,
    **params
) -> Tensor:
    """
    Conditional Pearson correlation of variables in a tensor batch.

    The correlation is obtained via normalisation of the conditional
    covariance. Given a conditional covariance matrix
    :math:`\hat{{\Sigma}} \in \mathbb{{R}}^{{n \times n}}`, each
    entry of the conditional correlation matrix
    :math:`R \in \mathbb{{R}}^{{n \times n}}`
    is defined according to

    :math:`R_{{ij}} = \frac{{\hat{{\Sigma}}_{{ij}}}}{{\sqrt{{\hat{{\Sigma}}_{{ii}}}} \sqrt{{\hat{{\Sigma}}_{{jj}}}}`

    :Dimension: {binary_dim_spec}

    Parameters
    ----------\
    {conditional_param_spec}\
    {inverting_param_spec}\
    {param_spec}

    Returns
    -------
    sigma : Tensor
        Conditional empirical covariance matrix of the variables in input
        tensor X conditioned on the variables in input tensor Y.

    See also
    --------
    conditionalcov: Conditional covariance (unnormalised)
    partialcorr: Condition each variable on all other variables
    """
    sigma = conditionalcov(
        X, Y, require_nonsingular=require_nonsingular, **params)
    fact = corrnorm(sigma)
    return sigma / fact


@document_covariance
def precision(
    X: Tensor,
    require_nonsingular: bool = True,
    **params
) -> Tensor:
    """
    Empirical precision of variables in a tensor batch.

    The precision matrix is the inverse of the covariance matrix.

    ..note::
        The precision matrix is not defined for singular covariance matrices.
        If the number of input observations is less than the number of
        variables, the covariance matrix can be regularised to ensure it is
        non-singular. This is done by setting the ``l2`` parameter to a
        positive value. Alternatively, the ``require_nonsingular`` parameter
        can be set to `False` to use the Moore-Penrose pseudoinverse of the
        covariance matrix.

    :Dimension: {unary_dim_spec}

    Parameters
    ----------\
    {unary_param_spec}\
    {inverting_param_spec}\
    {param_spec}

    Returns
    -------
    omega : Tensor
        Precision matrix of the variables in input tensor X.

    See also
    --------
    cov: Empirical covariance matrix
    partialcorr: Partial correlation matrix
    partialcov: Partial covariance matrix
    """
    sigma = cov(X, **params)
    if require_nonsingular:
        return jnp.linalg.inv(sigma)
    else:
        return jnp.linalg.pinv(sigma)


def corrnorm(A: Tensor) -> Tensor:
    """
    Normalisation term for the correlation coefficient.

    Parameters
    ----------
    A : Tensor
        Batch of covariance or unnormalised correlation matrices.

    Returns
    -------
    normfact: Tensor
        Normalisation term for each element of the input tensor. Dividing by
        this will yield the normalised correlation.
    """
    d = jnp.diagonal(A, axis1=-2, axis2=-1)
    fact = -jax.lax.sqrt(d)[..., None]
    return (fact @ fact.swapaxes(-1, -2) + jnp.finfo(fact.dtype).eps)


def covariance(*pparams, **params):
    """Alias for :func:`cov`."""
    return cov(*pparams, **params)


def correlation(*pparams, **params):
    """Alias for :func:`corr`."""
    return corr(*pparams, **params)


def corrcoef(*pparams, **params):
    """Alias for :func:`corr`."""
    return corr(*pparams, **params)


def pcorr(*pparams, **params):
    """Alias for :func:`partialcorr`."""
    return partialcorr(*pparams, **params)


def ccov(*pparams, **params):
    """Alias for :func:`conditionalcov`."""
    return conditionalcov(*pparams, **params)


def ccorr(*pparams, **params):
    """Alias for :func:`conditionalcorr`."""
    return conditionalcorr(*pparams, **params)


def _prepare_input(X: Tensor, rowvar: bool = True) -> Tensor:
    """
    Ensure that the input is conformant with the transposition expected by the
    covariance function.
    """
    X = jnp.atleast_2d(X)
    if not rowvar and X.shape[-2] != 1:
        X = X.swapaxes(-1, -2)
    return X


def _prepare_weight_and_avg(
    vars: Sequence[Tensor],
    weight: Optional[Tensor] = None
) -> Tuple[
    Optional[Tensor],
    Optional[Literal['vector', 'matrix']],
    Union[float, int],
    Sequence[Tensor]
]:
    """
    Set the weights for the covariance computation based on user input and
    determine the sum of weights for normalisation. If weights are not
    provided, the sum is simply the count of data observations. Compute the
    first moment or its weighted analogue.
    """
    if weight is not None:
        if weight.ndim == 1 or weight.shape[-1] != weight.shape[-2]:
            w_type = 'vector'
            weight = _conform_vector_weight(weight)
            w_sum = weight.sum(-1, keepdims=True)
            avg = [
                (V * (weight / w_sum)).sum(-1, keepdims=True) for V in vars
            ]
        else:
            w_type = 'matrix'
            w_sum = weight.sum((-1, -2), keepdims=True)
            #TODO
            # We'll need to ensure that this is correct
            # for the nondiagonal case. The tests still don't.
            avg = [
                (V @ (weight / w_sum)).sum(-1, keepdims=True) for V in vars
            ]
    else:
        w_type = None
        w_sum = vars[0].shape[-1]
        avg = [V.mean(-1, keepdims=True) for V in vars]
    return weight, w_type, w_sum, avg


def _prepare_denomfact(
    w_sum: Union[float, int],
    w_type: Optional[Literal['vector', 'matrix']] = 'matrix',
    ddof: Optional[int] = None,
    bias: bool = False,
    weight: Optional[Tensor] = None
) -> Tensor:
    """
    Determine the factor we should divide by to obtain the (un)biased
    covariance from the sum over observations.
    """
    if ddof is None: ddof = int(not bias)
    if weight is None:
        fact = w_sum - ddof
    elif ddof == 0:
        fact = w_sum
    elif w_type == 'vector':
        fact = w_sum - ddof * (weight ** 2).sum(-1, keepdims=True) / w_sum
    else:
        #TODO
        # I don't have the intuition here yet: should this be
        # weight * weight or weight @ weight ? Or something else?
        # This affects only the nondiagonal case.
        fact = w_sum - ddof * (
            #weight.sum(-1, keepdims=True) ** 2
            weight @ weight.swapaxes(-1, -2)
            ).sum((-1, -2), keepdims=True) / w_sum
    return fact
