# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Symmetric matrix powers
~~~~~~~~~~~~~~~~~~~~~~~
Differentiable computation of matrix logarithm, exponential, and square root.
For use with symmetric, positive semidefinite matrices.
"""
import torch


def symxfm(input, xfm, spd=True, psi=0):
    """
    Apply a specified matrix-valued transformation to a batch of symmetric
    (probably positive semidefinite) tensors.

    Dimension
    ---------
    - Input: :math:`(N, *, D, D)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      D denotes matrix row and column dimension
    - Output: :math:`(N, *, D, D)`

    Parameters
    ----------
    input : Tensor
        Batch of symmetric tensors to transform.
    xfm : dimension-conserving torch callable
        Transformation to apply as a matrix-valued function
    spd : bool (default True)
        Indicates that the matrices in the input batch are symmetric positive
        semidefinite; guards against numerical rounding errors and ensures all
        eigenvalues are nonnegative.
    psi: float in [0, 1]
        Conditioning factor to promote positive definiteness. If this is in
        (0, 1], the original input will be replaced with a convex combination
        of the input and an identity matrix.

        :math:`\widetilde{X} = (1 - \psi) X + \psi I`

        A suitable `psi` can be used to ensure that all eigenvalues are
        positive.

    Returns
    -------
    output : Tensor
        Transformation of each matrix in the input batch.
    """
    if psi > 0:
        if psi > 1:
            raise ValueError('Nonconvex combination. Select psi in [0, 1].')
        input = (1 - psi) * input + psi * torch.eye(input.size(-1))
    L, Q = torch.symeig(input, eigenvectors=True)
    if spd:
        L = torch.maximum(L, torch.zeros_like(L))
    Lxfm = torch.diag_embed(xfm(L))
    return Q @ Lxfm @ Q.transpose(-1, -2)


def symlog(input, recondition=0):
    """
    Matrix logarithm of a batch of symmetric, positive definite matrices.

    Computed by diagonalising the matrix, computing the logarithm of the
    eigenvalues, and recomposing.

    :math:`\log X = Q_X \log \Lambda_X Q_X^\intercal`

    Note that this will be infeasible for singular or non-positive definite
    matrices. To guard against the infeasible case, consider specifying a
    `recondition` parameter.

    Dimension
    ---------
    - Input: :math:`(N, *, D, D)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      D denotes matrix row and column dimension
    - Output: :math:`(N, *, D, D)`

    Parameters
    ----------
    input : Tensor
        Batch of symmetric tensors to transform using the matrix logarithm.
    recondition: float in [0, 1]
        Conditioning factor to promote positive definiteness. If this is in
        (0, 1], the original input will be replaced with a convex combination
        of the input and an identity matrix.

        :math:`\hat{X} = (1 - \psi) X + \psi I`

        A suitable value can be used to ensure that all eigenvalues are
        positive and therefore guarantee that the matrix is in the domain.

    Returns
    -------
    output : Tensor
        Logarithm of each matrix in the input batch.
    """
    return symxfm(input, torch.log, psi=recondition)


def symexp(input):
    """
    Matrix exponential of a batch of symmetric, positive definite matrices.

    Computed by diagonalising the matrix, computing the exponential of the
    eigenvalues, and recomposing.

    :math:`\exp X = Q_X \exp \Lambda_X Q_X^\intercal`

    Dimension
    ---------
    - Input: :math:`(N, *, D, D)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      D denotes matrix row and column dimension
    - Output: :math:`(N, *, D, D)`

    Parameters
    ----------
    input : Tensor
        Batch of symmetric tensors to transform using the matrix exponential.

    Returns
    -------
    output : Tensor
        Exponential of each matrix in the input batch.
    """
    return symxfm(input, torch.exp)


def symsqrt(input, recondition=0):
    """
    Matrix square root of a batch of symmetric, positive definite matrices.

    Computed by diagonalising the matrix, computing the square root of the
    eigenvalues, and recomposing.

    :math:`\sqrt{X} = Q_X \sqrt{\Lambda_X} Q_X^\intercal`

    Note that this will be infeasible for matrices with negative eigenvalues,
    and potentially singular matrices due to numerical rounding errors. To
    guard against the infeasible case, consider specifying a `recondition`
    parameter.

    Dimension
    ---------
    - Input: :math:`(N, *, D, D)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      D denotes matrix row and column dimension
    - Output: :math:`(N, *, D, D)`

    Parameters
    ----------
    input : Tensor
        Batch of symmetric tensors to transform using the matrix square root.
    recondition: float in [0, 1]
        Conditioning factor to promote positive definiteness. If this is in
        (0, 1], the original input will be replaced with a convex combination
        of the input and an identity matrix.

        :math:`\hat{X} = (1 - \psi) X + \psi I`

        A suitable value can be used to ensure that all eigenvalues are
        nonnegative and therefore guarantee that the matrix is in the domain.

    Returns
    -------
    output : Tensor
        Square root of each matrix in the input batch.
    """
    return symxfm(input, torch.sqrt, psi=recondition)
