# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Covariance
~~~~~~~~~~
Differentiable estimation of covariance and derived measures
"""
import torch


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
    weight, w_sum, (avg,) = _prepare_weight_and_avg((X,), weight)
    fact = _prepare_denomfact(w_sum, ddof, bias, weight)

    X0 = X - avg.expand_as(X)
    if weight is None:
        sigma = X0 @ X0.transpose(-1, -2) / fact
    else:
        sigma = X0 @ weight @ X0.transpose(-1, -2) / fact
    if l2 > 0:
        sigma += torch.eye(X.size(-2))
    return sigma


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
    weight, w_sum, (Xavg, Yavg) = _prepare_weight_and_avg((X, Y), weight)
    fact = _prepare_denomfact(w_sum, ddof, bias, weight)

    X0 = X - Xavg.expand_as(X)
    Y0 = Y - Yavg.expand_as(Y)
    if weight is None:
        sigma = X0 @ Y0.transpose(-1, -2) / fact
    else:
        sigma = X0 @ weight @ Y0.transpose(-1, -2) / fact
    return sigma


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
        if weight.dim() == 1:
            weight = torch.diagflat(weight)
        w_sum = weight.sum()
        #TODO
        # We'll need to ensure that this is correct
        # for the nondiagonal case. The tests still don't.
        for V in vars:
            avg += [(V @ (weight / w_sum)).sum(-1, keepdim=True)]
    else:
        w_sum = vars[0].size(-1)
        for V in vars:
            avg += [V.mean(-1, keepdim=True)]
    return weight, w_sum, avg


def _prepare_denomfact(w_sum, ddof=None, bias=False, weight=None):
    """
    Determine the factor we should divide by to obtain the (un)biased
    covariance from the sum over observations.
    """
    ddof = ddof or int(not bias)
    if weight is None:
        fact = w_sum - ddof
    elif ddof == 0:
        fact = w_sum
    else:
        #TODO
        # I don't have the intuition here yet: should this be
        # weight * weight or weight @ weight ? This affects only
        # the nondiagonal case.
        fact = w_sum - ddof * (weight @ weight).sum() / w_sum
    return fact
