# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Covariance
~~~~~~~~~~
Differentiable estimation of covariance and derived measures
"""
import torch
from .matrix import invert_spd


def cov(X, rowvar=True, bias=False, ddof=None, weight=None, l2=0):
    """
    Empirical covariance of variables in a tensor batch.
    Thanks to https://github.com/pytorch/pytorch/issues/19037 for a more
    complete implementation.

    Dimension
    ---------
    - Input: :math:`(N, *, C, obs)` or :math:`(N, *, obs, C)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      C denotes number of data channels or variables to be correlated
    - Weight: :math:`(obs)` or :math:`(obs, obs)`
    - Output: :math:`(N, *, C, C)`

    Parameters
    ----------
    X : Tensor
        Tensor containing a sample of multivariate observations. Each slice
        along the last axis corresponds to an observation, and each slice along
        the penultimate axis corresponds to a data channel or more generally a
        variable.
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
        or multi-diagonal tensor can be used to specify inter-temporal coupling
        for a time series covariance.
    l2: nonnegative float (default 0)
        L2 regularisation term to add to the maximum likelihood estimate of the
        covariance matrix. This can be set to a positive value to obtain an
        intermediate for estimating the regularised inverse covariance.

    Returns
    -------
    sigma: Tensor
        Empirical covariance matrix of the variables in the input tensor.

    See also
    --------
    corr: Normalised covariance matrix (Pearson correlation matrix)
    precision: Inverse covariance (precision) matrix
    partialcorr: Partial correlation matrix
    """
    X = _prepare_input(X, rowvar)
    weight, w_type, w_sum, (avg,) = _prepare_weight_and_avg((X,), weight)
    fact = _prepare_denomfact(w_sum, w_type, ddof, bias, weight)

    X0 = X - avg
    if weight is None:
        sigma = X0 @ X0.transpose(-1, -2) / fact
    elif w_type == 'vector':
        sigma = (X0 * weight) @ X0.transpose(-1, -2) / fact
    else:
        sigma = X0 @ weight @ X0.transpose(-1, -2) / fact
    if l2 > 0:
        sigma += l2 * torch.eye(X.size(-2))
    return sigma


def corr(X, **params):
    r"""
    Pearson correlation of variables in a tensor batch.

    Consult the `cov` documentation for complete parameter characteristics.
    The correlation is obtained via normalisation of the covariance. Given a
    covariance matrix :math:`\hat{\Sigma} \in \mathbb{R}^{n \times n}`, each
    entry of the correlation matrix :math:`R \in \mathbb{R}^{n \times n}` is
    defined according to

    :math:`R_{ij} = \frac{\hat{\Sigma}_{ij}}{\hat{\Sigma}_{ii} \hat{\Sigma}_{ij}}`
    """
    sigma = cov(X, **params)
    fact = corrnorm(sigma)
    return sigma / fact


def partialcov(X, **params):
    """
    Partial covariance of variables in a tensor batch.

    Consult the `cov` documentation for complete parameter characteristics. The
    partial covariance is obtained by conditioning the covariance of each pair
    of variables on all other observed variables. It can be interpreted as a
    measurement of the direct relationship between each variable pair. The
    partial covariance is computed via inversion of the covariance matrix,
    followed by negation of off-diagonal entries.
    """
    omega = precision(X, **params)
    #TODO: let's add a test to make sure this in-place op doesn't mess up the
    # computational graph
    omega[..., ~torch.eye(omega.size(-1), dtype=torch.bool)] *= -1
    return omega


def partialcorr(X, **params):
    """
    Partial Pearson correlation of variables in a tensor batch.

    Consult the `cov` documentation for complete parameter characteristics. The
    partial correlation is obtained by conditioning the covariance of each pair
    of variables on all other observed variables. It can be interpreted as a
    measurement of the direct relationship between each variable pair. The
    partial correlation is efficiently computed via successive inversion and
    normalisation of the covariance matrix, accompanied by negation of off-
    diagonal entries.
    """
    omega = partialcov(X, **params)
    fact = corrnorm(omega)
    return omega / fact


def pairedcov(X, Y, rowvar=True, bias=False, ddof=None, weight=None, l2=0):
    """
    Empirical covariance between two sets of variables.

    This function does not offer any performance improvement relative to
    computing the complete covariance matrix of all variables.

    Dimension
    ---------
    - Input X: :math:`(N, *, C_X, obs)` or :math:`(N, *, obs, C_X)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      C_X denotes number of data channels or variables to be correlated with
      those in input Y
    - Input Y: :math:`(N, *, C_Y, obs)` or :math:`(N, *, obs, C_Y)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      C_Y denotes number of data channels or variables to be correlated with
      those in input X
    - Weight: :math:`(obs)` or :math:`(obs, obs)`
    - Output: :math:`(N, *, C_X, C_Y)`

    Parameters
    ----------
    X, Y : Tensors
        Tensors containing samples of multivariate observations. Each slice
        along the last axis corresponds to an observation, and each slice along
        the penultimate axis corresponds to a data channel or more generally a
        variable.
    rowvar, bias, ddof, weight
        Consult the `cov` documentation for complete parameter characteristics.
    l2
        Has no effect. Included for conformance with `cov`.
    """
    X = _prepare_input(X, rowvar)
    Y = _prepare_input(Y, rowvar)
    weight, w_type, w_sum, (Xavg, Yavg) = _prepare_weight_and_avg((X, Y), weight)
    fact = _prepare_denomfact(w_sum, w_type, ddof, bias, weight)

    X0 = X - Xavg
    Y0 = Y - Yavg
    if weight is None:
        sigma = X0 @ Y0.transpose(-1, -2) / fact
    elif w_type == 'vector':
        sigma = (X0 * weight) @ Y0.transpose(-1, -2) / fact
    else:
        sigma = X0 @ weight @ Y0.transpose(-1, -2) / fact
    return sigma


def pairedcorr(X, Y, **params):
    r"""
    Empirical Pearson correlation between variables in two tensor batches.

    Consult the `pairedcov`documentation for complete parameter details.
    The empirical paired correlation is obtained via normalisation of the
    paired covariance. Given a paired covariance matrix
    :math:`\hat{\Sigma} \in \mathbb{R}^{n \times n}`, each entry of the
    paired correlation matrix :math:`R \in \mathbb{R}^{n \times n}`
    is defined according to

    :math:`R_{ij} = \frac{\hat{\Sigma}_{ij}}{\hat{\Sigma}_{ii} \hat{\Sigma}_{ij}}`
    """
    varX, varY = torch.var(X, -1, keepdim=True), torch.var(Y, -1, keepdim=True)
    fact = torch.sqrt(varX @ varY.transpose(-2, -1))
    return pairedcov(X, Y, **params) / fact


def conditionalcov(X, Y, **params):
    r"""
    Conditional covariance of variables in a tensor batch.

    The conditional covariance is the covariance of one set of variables X
    conditioned on another set of variables Y. The conditional covariance is
    computed from the covariance A of X , the covariance C of Y, and the
    covariance B between X and Y. It is defined as the Schur complement of C:

    :math:`\widetilde{Sigma} = A - B C^{-1} B^\intercal`

    The conditional covariance is equivalent to the covariance of the first set
    of variables after residualising them with respect to the second set of
    variables (plus an intercept term). This can be interpreted as the
    covariance of variables of interest (the first set) after controlling for
    the effects of confounds or nuisance variables (the second set).

    Dimension
    ---------
    - Input X: :math:`(N, *, C_X, obs)` or :math:`(N, *, obs, C_X)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      C_X denotes number of data channels or variables
    - Input Y: :math:`(N, *, C_Y, obs)` or :math:`(N, *, obs, C_Y)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      C_Y denotes number of data channels or variables
    - Weight: :math:`(obs)` or :math:`(obs, obs)`
    - Output: :math:`(N, *, C_X, C_X)`

    Parameters
    ----------
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
        generally a variable.
    rowvar, bias, ddof, weight, l2
        Consult the `cov` documentation for complete parameter characteristics.

    Returns
    -------
    sigma: Tensor
        Conditional empirical covariance matrix of the variables in input
        tensor X conditioned on the variables in input tensor Y.

    See also
    --------
    conditionalcorr: Normalised conditional covariance (Pearson correlation)
    partialcorr: Condition each variable on all other variables
    """
    A = cov(X, **params)
    B = pairedcov(X, Y, **params)
    C = cov(Y, **params)
    return A - B @ invert_spd(C) @ B.transpose(-1, -2)


def conditionalcorr(X, Y, **params):
    r"""
    Conditional Pearson correlation of variables in a tensor batch.

    Consult the `conditionalcov` and `cov` documentation for complete parameter
    characteristics. The conditional correlation is obtained via normalisation
    of the conditional covariance. Given a conditional covariance matrix
    :math:`\hat{\Sigma} \in \mathbb{R}^{n \times n}`, each entry of the
    conditional correlation matrix :math:`R \in \mathbb{R}^{n \times n}`
    is defined according to

    :math:`R_{ij} = \frac{\hat{\Sigma}_{ij}}{\hat{\Sigma}_{ii} \hat{\Sigma}_{ij}}`
    """
    sigma = conditionalcov(X, Y, **params)
    fact = corrnorm(sigma)
    return sigma / fact


def precision(X, **params):
    """
    Empirical precision of variables in a tensor batch.

    The precision matrix is the inverse of the covariance matrix. Parameters
    available for covariance estimation are thus also available for precision
    estimation. Consult the `cov` documentation for complete parameter
    characteristics.
    """
    sigma = cov(X, **params)
    return invert_spd(sigma)


def corrnorm(A):
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
    d = torch.diagonal(A, dim1=-2, dim2=-1)
    fact = -torch.sqrt(d).unsqueeze(-1)
    return (fact @ fact.transpose(-1, -2))


def covariance(*pparams, **params):
    """Alias for `cov`."""
    return cov(*pparams, **params)


def correlation(*pparams, **params):
    """Alias for `corr`."""
    return corr(*pparams, **params)


def corrcoef(*pparams, **params):
    """Alias for `corr`."""
    return corr(*pparams, **params)


def pcorr(*pparams, **params):
    """Alias for `partialcorr`."""
    return partialcorr(*pparams, **params)


def ccov(*pparams, **params):
    """Alias for `conditionalcov`."""
    return conditionalcov(*pparams, **params)


def ccorr(*pparams, **params):
    """Alias for `conditionalcorr`."""
    return conditionalcorr(*pparams, **params)


def _prepare_input(X, rowvar=True):
    """
    Ensure that the input is conformant with the transposition expected by the
    covariance function.
    """
    if X.dim() == 1:
        X = X.view(1, -1)
    if not rowvar and X.size(-2) != 1:
        X = X.transpose(-1, -2)
    return X


def _prepare_weight_and_avg(vars, weight=None):
    """
    Set the weights for the covariance computation based on user input and
    determine the sum of weights for normalisation. If weights are not
    provided, the sum is simply the count of data observations. Compute the
    first moment or its weighted analogue.
    """
    avg = []
    if weight is not None:
        if weight.dim() == 1 or weight.shape[-1] != weight.shape[-2]:
            w_type = 'vector'
            weight = _conform_vector_weight(weight)
            w_sum = weight.sum(-1, keepdim=True)
            for V in vars:
                avg += [(V * (weight / w_sum)).sum(-1, keepdim=True)]
        else:
            w_type = 'matrix'
            w_sum = weight.sum([-1, -2], keepdim=True)
            #TODO
            # We'll need to ensure that this is correct
            # for the nondiagonal case. The tests still don't.
            for V in vars:
                avg += [(V @ (weight / w_sum)).sum(-1, keepdim=True)]
    else:
        w_type = None
        w_sum = vars[0].size(-1)
        for V in vars:
            avg += [V.mean(-1, keepdim=True)]
    return weight, w_type, w_sum, avg


def _conform_vector_weight(weight):
    if weight.dim() == 1:
        return weight
    if weight.shape[-2] != 1:
        return weight.unsqueeze(-2)
    return weight


def _prepare_denomfact(w_sum, w_type='matrix', ddof=None,
                       bias=False, weight=None):
    """
    Determine the factor we should divide by to obtain the (un)biased
    covariance from the sum over observations.
    """
    ddof = ddof or int(not bias)
    if weight is None:
        fact = w_sum - ddof
    elif ddof == 0:
        fact = w_sum
    elif w_type == 'vector':
        fact = w_sum - ddof * (weight ** 2).sum(-1, keepdim=True) / w_sum
    else:
        #TODO
        # I don't have the intuition here yet: should this be
        # weight * weight or weight @ weight ? This affects only
        # the nondiagonal case.
        fact = w_sum - ddof * (weight @ weight.transpose(-1, -2)).sum(
            [-1, -2], keepdim=True) / w_sum
    return fact