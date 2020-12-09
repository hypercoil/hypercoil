# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Crosshair-kernel similarity
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Localised similarity functions over a crosshair kernel.
"""
import torch
from .crosshair import (
    crosshair_dot,
    crosshair_norm_l1,
    crosshair_norm_l2
)
from .matrix import expand_outer


def crosshair_similarity(A, L, R=None, symmetry=None):
    """
    Use a crosshair kernel to assess the local similarity between a
    matrix A and low-rank templates L x R transpose.

    The default crosshair similarity function defines similarity as a simple
    dot product between the ravelled crosshair vectors.

    Note that the output of this procedure is a matrix approximately of
    rank 1 (first singular value will dominate variance).

    Parameters
    ----------
    A: Tensor (N x C_in x H x W)
        The matrix in which features should be identified.
    L: Tensor (C_out x C_in x H x rank)
        Left generator of a low-rank matrix (L x R transpose) that specifies
        a feature set to find in A.
    R: Tensor or None (C_out x C_in x W x rank)
        Right generator of a low-rank matrix (L x R transpose) that specifies
        a feature set to find in A. If this is None, then the template matrix
        is symmetric L x L transpose.
    symmetry: 'cross', 'skew', or other
        Symmetry constraint imposed on the generated low-rank template matrix.
        * `cross` enforces symmetry by replacing the initial expansion with
          the average of the initial expansion and its transpose.
        * `skew` enforcesd skew-symmetry by subtracting from the initial
          expansion its transpose.
        * Otherwise, no explicit symmetry constraint is imposed. Symmetry can
          also be enforced by passing None for R or by passing the same input
          for R and L.
    """
    B = expand_outer(L, R, symmetry)
    return crosshair_dot(A.unsqueeze(1), B).sum(2)


def crosshair_cosine_similarity(A, L, R=None, symmetry=None):
    """
    Use a crosshair kernel to assess the local similarity between a
    matrix A and low-rank templates L x R transpose.

    The crosshair cosine similarity function defines similarity as the cosine
    similarity between the ravelled crosshair vectors.

    Note that the output of this procedure is a matrix approximately of
    rank 1 (first singular value will dominate variance).

    Parameters
    ----------
    A: Tensor (N x C_in x H x W)
        The matrix in which features should be identified.
    L: Tensor (C_out x C_in x H x rank)
        Left generator of a low-rank matrix (L x R transpose) that specifies
        a feature set to find in A.
    R: Tensor or None (C_out x C_in x W x rank)
        Right generator of a low-rank matrix (L x R transpose) that specifies
        a feature set to find in A. If this is None, then the template matrix
        is symmetric L x L transpose.
    symmetry: 'cross', 'skew', or other
        Symmetry constraint imposed on the generated low-rank template matrix.
        * `cross` enforces symmetry by replacing the initial expansion with
          the average of the initial expansion and its transpose.
        * `skew` enforcesd skew-symmetry by subtracting from the initial
          expansion its transpose.
        * Otherwise, no explicit symmetry constraint is imposed. Symmetry can
          also be enforced by passing None for R or by passing the same input
          for R and L.
    """
    B = expand_outer(L, R, symmetry)
    num = crosshair_dot(A.unsqueeze(1), B)
    denom0 = crosshair_norm_l2(A.unsqueeze(1))
    denom1 = crosshair_norm_l2(B)
    return (num / (denom0 * denom1)).sum(2)


def crosshair_l1_similarity(A, L, R=None, symmetry=None):
    """
    Use a crosshair kernel to assess the local similarity between a
    matrix A and low-rank templates L x R transpose.

    The crosshair L1 similarity function defines similarity as the L1 distance
    between the ravelled crosshair vectors.

    Note that the output of this procedure is a matrix approximately of
    rank 1 (first singular value will dominate variance).

    Parameters
    ----------
    A: Tensor (N x C_in x H x W)
        The matrix in which features should be identified.
    L: Tensor (C_out x C_in x H x rank)
        Left generator of a low-rank matrix (L x R transpose) that specifies
        a feature set to find in A.
    R: Tensor or None (C_out x C_in x W x rank)
        Right generator of a low-rank matrix (L x R transpose) that specifies
        a feature set to find in A. If this is None, then the template matrix
        is symmetric L x L transpose.
    symmetry: 'cross', 'skew', or other
        Symmetry constraint imposed on the generated low-rank template matrix.
        * `cross` enforces symmetry by replacing the initial expansion with
          the average of the initial expansion and its transpose.
        * `skew` enforcesd skew-symmetry by subtracting from the initial
          expansion its transpose.
        * Otherwise, no explicit symmetry constraint is imposed. Symmetry can
          also be enforced by passing None for R or by passing the same input
          for R and L.
    """
    B = expand_outer(L, R, symmetry)
    diff = A.unsqueeze(1) - B
    return crosshair_norm_l1(diff).sum(2)


def crosshair_l2_similarity(A, L, R=None, symmetry=None):
    """
    Use a crosshair kernel to assess the local similarity between a
    matrix A and low-rank templates L x R transpose.

    The crosshair L2 similarity function defines similarity as the L2 distance
    between the ravelled crosshair vectors.

    Note that the output of this procedure is a matrix approximately of
    rank 1 (first singular value will dominate variance).

    Parameters
    ----------
    A: Tensor (N x C_in x H x W)
        The matrix in which features should be identified.
    L: Tensor (C_out x C_in x H x rank)
        Left generator of a low-rank matrix (L x R transpose) that specifies
        a feature set to find in A.
    R: Tensor or None (C_out x C_in x W x rank)
        Right generator of a low-rank matrix (L x R transpose) that specifies
        a feature set to find in A. If this is None, then the template matrix
        is symmetric L x L transpose.
    symmetry: 'cross', 'skew', or other
        Symmetry constraint imposed on the generated low-rank template matrix.
        * `cross` enforces symmetry by replacing the initial expansion with
          the average of the initial expansion and its transpose.
        * `skew` enforcesd skew-symmetry by subtracting from the initial
          expansion its transpose.
        * Otherwise, no explicit symmetry constraint is imposed. Symmetry can
          also be enforced by passing None for R or by passing the same input
          for R and L.
    """
    B = expand_outer(L, R, symmetry)
    diff = A.unsqueeze(1) - B
    return crosshair_norm_l2(diff).sum(2)
