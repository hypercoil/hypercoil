# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Residualise tensor block via least squares.
"""
import jax.numpy as jnp
from hypercoil.functional.utils import Tensor, vmap_over_outer


def residualise(
    Y: Tensor,
    X: Tensor,
    rowvar: bool = True
) -> Tensor:
    r"""
    Residualise a tensor block via ordinary linear least squares.

    .. warning::
        In some cases, we have found that the least-squares fit returned is
        incorrect for reasons that are not clear. (Incorrect results are
        returned by
        ``torch.linalg.lstsq``, although correct results are returned if
        ``torch.linalg.pinv`` is used instead.) Verify that results are
        reasonable when using this operation.

    .. note::
        The :doc:`conditional covariance <hypercoil.functional.cov.conditionalcov>`
        or :doc:`conditional correlation <hypercoil.functional.cov.conditionalcorr>`
        may be used instead where appropriate.

    :Dimension: **Input Y :** :math:`(N, *, C_Y, obs)` or :math:`(N, *, obs, C_Y)`
                    N denotes batch size, `*` denotes any number of
                    intervening dimensions, :math:`C_Y` denotes number of data
                    channels or variables, obs denotes number of observations
                    per channel
                **Input X :** :math:`(N, *, C_X, obs)` or :math:`(N, *, obs, C_X)`
                    :math:`C_X` denotes number of data channels or variables
                **Output :**  :math:`(N, *, C_Y, obs)` or :math:`(N, *, obs, C_Y)`
                    As above.

    Parameters
    ----------
    Y : Tensor
        Tensor to be residualised or orthogonalised with respect to `X`. The
        vector of observations in each channel is projected into a subspace
        orthogonal to the span of `X`.
    X : Tensor
        Tensor containing explanatory variables. Any variance in `Y` that can
        be explained by variables in `X` will be removed from `Y`.
    driver : str (default `'gelsd'`)
        Driver routine for solving linear least squares. See LAPACK
        documentation for further details.
    rowvar : bool (default True)
        Indicates that the last axis of the input tensor is the observation
        axis and the penultimate axis is the variable axis. If False, then this
        relationship is transposed.
    """
    if rowvar:
        X_in = X.swapaxes(-1, -2)
        Y_in = Y.swapaxes(-1, -2)
    else:
        X_in, Y_in = X, Y
    fit = vmap_over_outer(jnp.linalg.lstsq, 2)
    betas = fit((X_in, Y_in))[0]
    if rowvar:
        return Y - betas.swapaxes(-1, -2) @ X
    return Y - X @ betas
