# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Differentiable computation of matrix logarithm, exponential, and square root.
For use with symmetric (typically positive semidefinite) matrices.
"""
import torch
from . import symmetric, recondition_eigenspaces


#TODO: Look here more closely:
# https://github.com/pytorch/pytorch/issues/25481
# regarding limitations and more efficient implementations


#TODO: Let's think about overhauling SPD operations using
# https://geoopt.readthedocs.io/en/latest/index.html
# We could also ideally replace the domain mapper system with
# manifold-aware optimisers.


def symmap(input, map, spd=True, psi=0,
           recondition='eigenspaces',
           truncate_eigenvalues=False):
    r"""
    Apply a specified matrix-valued transformation to a batch of symmetric
    (probably positive semidefinite) tensors.

    :Dimension: **Input :** :math:`(N, *, D, D)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, D denotes matrix row and column
                    dimension.
                **Output :** :math:`(N, *, D, D)`
                    As above.

    Parameters
    ----------
    input : Tensor
        Batch of symmetric tensors to transform.
    map : dimension-conserving torch callable
        Transformation to apply as a matrix-valued function.
    spd : bool (default True)
        Indicates that the matrices in the input batch are symmetric positive
        semidefinite; guards against numerical rounding errors and ensures all
        eigenvalues are nonnegative.
    psi : float in [0, 1]
        Conditioning factor to promote positive definiteness.
    recondition : ``'convexcombination'`` or ``'eigenspaces'`` (default ``'eigenspaces'``)
        Method for reconditioning.

        - ``'convexcombination'`` denotes that the original input will be
          replaced with a convex combination of the input and an identity
          matrix.

          :math:`\widetilde{X} = (1 - \psi) X + \psi I`

          A suitable :math:`\psi` can be used to ensure that all eigenvalues
          are positive.

        - ``'eigenspaces'`` denotes that noise will be added to the original
          input along the diagonal.

          :math:`\widetilde{X} = X + \psi I - \xi I`

          where each element of :math:`\xi` is independently sampled uniformly
          from :math:`(0, \psi)`. In addition to promoting positive
          definiteness, this method promotes eigenspaces with dimension 1 (no
          degenerate/repeated eigenvalues). Nondegeneracy of eigenvalues is
          required for differentiation through SVD.
    truncate_eigenvalues : bool (default False)
        Indicates that very small eigenvalues, which might for instance occur
        due to numerical errors in the decomposition, should be truncated to
        zero. Note that you should not do this if you wish to differentiate
        through this operation, or if you require the input to be positive
        definite. For these use cases, consider using the ``psi`` and
        ``recondition`` parameters.

    Returns
    -------
    output : Tensor
        Transformation of each matrix in the input batch.
    """
    if psi > 0:
        if psi > 1:
            raise ValueError('Nonconvex combination. Select psi in [0, 1].')
        if recondition == 'convexcombination':
            input = (1 - psi) * input + psi * torch.eye(
                input.size(-1), dtype=input.dtype, device=input.device
            )
        elif recondition == 'eigenspaces':
            input = recondition_eigenspaces(input, psi=psi, xi=psi)
    if not spd:
        Q, L, _ = torch.svd(symmetric(input))
    else:
        L, Q = torch.linalg.eigh(symmetric(input))
    if truncate_eigenvalues:
        # Based on xmodar's implementation here:
        # https://github.com/pytorch/pytorch/issues/25481
        above_cutoff = L > L.amax() * L.size(-1) * torch.finfo(L.dtype).eps
        L = L[..., above_cutoff]
        Q = Q[..., above_cutoff]
    Lmap = torch.diag_embed(map(L))
    return symmetric(Q @ Lmap @ Q.transpose(-1, -2))


#TODO: change symlog and symsqrt to support either reconditioning method.
def symlog(input, recondition=0):
    r"""
    Matrix logarithm of a batch of symmetric, positive definite matrices.

    Computed by diagonalising the matrix :math:`X = Q_X \Lambda_X Q_X^\intercal`,
    computing the logarithm of the eigenvalues, and recomposing.

    :math:`\log X = Q_X \log \Lambda_X Q_X^\intercal`

    Note that this will be infeasible for singular or non-positive definite
    matrices. To guard against the infeasible case, consider specifying a
    ``recondition`` parameter.

    :Dimension: **Input :** :math:`(N, *, D, D)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, D denotes matrix row and column
                    dimension.
                **Output :** :math:`(N, *, D, D)`
                    As above.

    Parameters
    ----------
    input : Tensor
        Batch of symmetric tensors to transform using the matrix logarithm.
    recondition : float in [0, 1]
        Conditioning factor to promote positive definiteness and nondegenerate
        eigenvalues. If this is in (0, 1], the original input will be replaced
        with

        :math:`\widetilde{X} = X + \psi I - \xi I`

        where each element of :math:`\xi` is independently sampled uniformly
        from :math:`(0, \psi)`. A suitable value can be used to ensure that
        all eigenvalues are positive and therefore guarantee that the matrix
        is in the domain.

    Returns
    -------
    output : Tensor
        Logarithm of each matrix in the input batch.
    """
    return symmap(input, torch.log, psi=recondition)


def symexp(input):
    r"""
    Matrix exponential of a batch of symmetric, positive definite matrices.

    Computed by diagonalising the matrix :math:`X = Q_X \Lambda_X Q_X^\intercal`,
    computing the exponential of the eigenvalues, and recomposing.

    :math:`\exp X = Q_X \exp \Lambda_X Q_X^\intercal`

    :Dimension: **Input :** :math:`(N, *, D, D)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, D denotes matrix row and column
                    dimension.
                **Output :** :math:`(N, *, D, D)`
                    As above.

    Parameters
    ----------
    input : Tensor
        Batch of symmetric tensors to transform using the matrix exponential.

    Returns
    -------
    output : Tensor
        Exponential of each matrix in the input batch.

    Warnings
    --------
    `torch.matrix_exp` is generally more stable and recommended over this.
    """
    return symmap(input, torch.exp)


def symsqrt(input, recondition=0):
    r"""
    Matrix square root of a batch of symmetric, positive definite matrices.

    Computed by diagonalising the matrix :math:`X = Q_X \Lambda_X Q_X^\intercal`,
    computing the square root of the eigenvalues, and recomposing.

    :math:`\sqrt{X} = Q_X \sqrt{\Lambda_X} Q_X^\intercal`

    Note that this will be infeasible for matrices with negative eigenvalues,
    and potentially singular matrices due to numerical rounding errors. To
    guard against the infeasible case, consider specifying a ``recondition``
    parameter.

    :Dimension: **Input :** :math:`(N, *, D, D)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, D denotes matrix row and column
                    dimension.
                **Output :** :math:`(N, *, D, D)`
                    As above.

    Parameters
    ----------
    input : Tensor
        Batch of symmetric tensors to transform using the matrix square root.
    recondition : float in [0, 1]
    recondition : float in [0, 1]
        Conditioning factor to promote positive definiteness and nondegenerate
        eigenvalues. If this is in (0, 1], the original input will be replaced
        with

        :math:`\widetilde{X} = X + \psi I - \xi I`

        where each element of :math:`\xi` is independently sampled uniformly
        from :math:`(0, \psi)`. A suitable value can be used to ensure that
        all eigenvalues are positive and therefore guarantee that the matrix
        is in the domain.

    Returns
    -------
    output : Tensor
        Square root of each matrix in the input batch.
    """
    return symmap(input, torch.sqrt, psi=recondition)
