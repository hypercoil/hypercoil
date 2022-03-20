# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Log determinant
~~~~~~~~~~~~~~~
Loss functions using the determinant or log determinant.
"""
import torch
from functools import partial
from .base import ReducingLoss
from ..functional import corr
from ..functional.matrix import recondition_eigenspaces


def log_det_corr(X, psi=0.001, xi=0.0099, cor=corr):
    """
    Compute the negative log determinant of the correlation matrix of the
    specified variables.

    Pearson correlation matrices are positive semidefinite and have a
    determinant in the range [0, 1], with a determinant of 1 corresponding
    to the case of orthogonal observations.

    Log determinant is a concave function, so finding a minimum for its
    negation should be straightforward.

    Parameters
    ----------
    X : tensor
    psi : float
        Reconditioning parameter for ensuring nonzero eigenvalues.
    xi : float
        Reconditioning parameter for ensuring nondegenerate eigenvalues.
    cor : callable (default corr)
        Covariance measure. By default, this is the Pearson correlation.
    """
    Z = cor(X)
    Z = recondition_eigenspaces(Z, psi=psi, xi=xi)
    return -torch.logdet(Z)


def det_corr(X, psi=0.001, xi=0.0099, cor=corr):
    """
    Compute the negative determinant of the correlation matrix of the
    specified variables.

    Pearson correlation matrices are positive semidefinite and have a
    determinant in the range [0, 1], with a determinant of 1 corresponding
    to the case of orthogonal observations.

    Parameters
    ----------
    X : tensor
    psi : float
        Reconditioning parameter for ensuring nonzero eigenvalues.
    xi : float
        Reconditioning parameter for ensuring nondegenerate eigenvalues.
    cor : callable (default corr)
        Covariance measure. By default, this is the Pearson correlation.
    """
    Z = cor(X)
    Z = recondition_eigenspaces(Z, psi=psi, xi=xi)
    return -torch.linalg.det(Z)


class LogDetCorr(ReducingLoss):
    """
    Compute the correlation matrix, and then penalise its negative log
    determinant.
    """
    def __init__(self, nu=1, psi=0.001, xi=0.001,
                 cor=corr, reduction=None, name=None):
        reduction = reduction or torch.mean
        self.psi = psi
        self.xi = xi
        logdetcorr = partial(log_det_corr, psi=self.psi, xi=self.xi, cor=cor)
        super(LogDetCorr, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=logdetcorr,
            name=name
        )


class DetCorr(ReducingLoss):
    """
    Compute the correlation matrix, and then penalise its negative
    determinant.
    """
    def __init__(self, nu=1, psi=0.001, xi=0.001,
                 cor=corr, reduction=None, name=None):
        reduction = reduction or torch.mean
        self.psi = psi
        self.xi = xi
        detcorr = partial(det_corr, psi=self.psi, xi=self.xi, cor=cor)
        super(DetCorr, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=logdetcorr,
            name=name
        )


class LogDet(ReducingLoss):
    """
    Penalise the negative log determinant, presumably of a positive definite
    matrix.

    Log determinant is a concave function, so finding a minimum for its
    negation should be straightforward.
    """
    def __init__(self, nu=1, reduction=None, name=None):
        reduction = reduction or torch.mean
        logdet = lambda X: -torch.logdet(X)
        super(LogDet, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=logdet,
            name=name
        )


class Determinant(ReducingLoss):
    def __init__(self, nu=1, reduction=None, name=None):
        reduction = reduction or torch.mean
        det = lambda X: -torch.linalg.det(X)
        super(Determinant, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=det,
            name=name
        )
