# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Residualise tensor block via least squares.
"""
import torch


def residualise(Y, X, driver='gelsd', rowvar=True):
    """
    Residualise a tensor block via ordinary linear least squares.

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

    Use `conditionalcorr` if possible.
    """
    if rowvar:
        X_in = X.transpose(-1, -2)
        Y_in = Y.transpose(-1, -2)
    else:
        X_in, Y_in = X, Y
    betas, _, _, _ = torch.linalg.lstsq(
        X_in, Y_in, driver=driver
    )
    if rowvar:
        return Y - betas.transpose(-1, -2) @ X
    return Y - X @ betas
