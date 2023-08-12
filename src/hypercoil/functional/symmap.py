# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Differentiable computation of matrix logarithm, exponential, and square root.
For use with symmetric (typically positive semidefinite) matrices.
"""
from __future__ import annotations
from typing import Callable, Literal, Optional

import jax
import jax.numpy as jnp

from ..engine import NestedDocParse, Tensor
from . import recondition_eigenspaces, symmetric


# TODO: Look here more closely:
# https://github.com/pytorch/pytorch/issues/25481
# regarding limitations and more efficient implementations


# TODO: Let's think about overhauling SPD operations using
# https://geoopt.readthedocs.io/en/latest/index.html
# We could also ideally replace the domain mapper system with
# manifold-aware optimisers.
# Now that we're switching to JAX as the backend, we can instead look into
# projections as in jaxopt. Not for gradients in our case, though.


def document_symmetric_map(func):
    symmap_param_spec = r"""
    psi : float in [0, 1]
        Conditioning factor to promote positive definiteness.
    key: Tensor or None (default None)
        Key for pseudo-random number generation. Required if ``recondition`` is
        set to ``'eigenspaces'`` and ``psi`` is in (0, 1].
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
    fill_nans : bool (default True)
        Indicates that any NaNs among the transformed eigenvalues should be
        replaced with zeros.
    truncate_eigenvalues : bool (default False)
        Indicates that very small eigenvalues, which might for instance occur
        due to numerical errors in the decomposition, should be truncated to
        zero. Note that you should not do this if you wish to differentiate
        through this operation, or if you require the input to be positive
        definite. For these use cases, consider using the ``psi`` and
        ``recondition`` parameters.
    """
    fmt = NestedDocParse(
        symmap_param_spec=symmap_param_spec,
    )
    func.__doc__ = func.__doc__.format_map(fmt)
    return func


@document_symmetric_map
def symmap(
    input: Tensor,
    map: Callable,
    spd: bool = True,
    psi: float = 0,
    key: Optional[Tensor] = None,
    recondition: Literal["eigenspaces", "convexcombination"] = "eigenspaces",
    fill_nans: bool = True,
    truncate_eigenvalues: bool = False,
) -> Tensor:
    r"""
    Apply a specified matrix-valued transformation to a batch of symmetric
    (probably positive semidefinite) tensors.

    .. note::
        This should be faster than using ``jax.scipy.linalg.funm`` for
        Hermitian matrices, although it is less general and probably less
        stable. This method relies on the eigendecomposition of the matrix.

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
    map : dimension-conserving callable
        Transformation to apply as a matrix-valued function.
    spd : bool (default True)
        Indicates that the matrices in the input batch are symmetric positive
        semidefinite; guards against numerical rounding errors and ensures all
        eigenvalues are nonnegative.\
    {symmap_param_spec}

    Returns
    -------
    output : Tensor
        Transformation of each matrix in the input batch.
    """
    if psi > 0:
        if psi > 1:
            raise ValueError("Nonconvex combination. Select psi in [0, 1].")
        if recondition == "convexcombination":
            input = (1 - psi) * input + psi * jnp.eye(input.shape[-1])
        elif recondition == "eigenspaces":
            input = recondition_eigenspaces(input, psi=psi, xi=psi, key=key)
    if not spd:
        return jax.scipy.linalg.funm(input, map)
    else:
        L, Q = jnp.linalg.eigh(symmetric(input))
    if truncate_eigenvalues:
        # Based on xmodar's implementation here:
        # https://github.com/pytorch/pytorch/issues/25481
        mask = (
            L > L.max(-1, keepdims=True) * L.shape[-1] * jnp.finfo(L.dtype).eps
        )
        L = jnp.where(mask, L, 0)
    Lmap = map(L)
    if fill_nans:
        Lmap = jnp.where(jnp.isnan(Lmap), 0, Lmap)
    return symmetric(Q @ (Lmap[..., None] * Q.swapaxes(-1, -2)))


@document_symmetric_map
def symlog(
    input: Tensor,
    psi: float = 0,
    key: Optional[Tensor] = None,
    recondition: Literal["eigenspaces", "convexcombination"] = "eigenspaces",
    fill_nans: bool = True,
    truncate_eigenvalues: bool = False,
) -> Tensor:
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
        Batch of symmetric tensors to transform using the matrix logarithm.\
    {symmap_param_spec}

    Returns
    -------
    output : Tensor
        Logarithm of each matrix in the input batch.
    """
    return symmap(
        input,
        jnp.log,
        psi=psi,
        key=key,
        recondition=recondition,
        fill_nans=fill_nans,
        truncate_eigenvalues=truncate_eigenvalues,
    )


def symexp(input: Tensor) -> Tensor:
    r"""
    Matrix exponential of a batch of symmetric, positive definite matrices.

    Computed by diagonalising the matrix :math:`X = Q_X \Lambda_X Q_X^\intercal`,
    computing the exponential of the eigenvalues, and recomposing.

    :math:`\exp X = Q_X \exp \Lambda_X Q_X^\intercal`

    .. note::
        This approach is in principle faster than the matrix exponential in
        JAX, but it is not as robust or general as the JAX implementation
        (``jax.linalg.expm``).

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
    ``jax.scipy.linalg.expm`` is generally more stable and recommended over this.
    """
    return symmap(input, jnp.exp)


@document_symmetric_map
def symsqrt(
    input: Tensor,
    psi: float = 0,
    key: Optional[Tensor] = None,
    recondition: Literal["eigenspaces", "convexcombination"] = "eigenspaces",
    fill_nans: bool = True,
    truncate_eigenvalues: bool = False,
) -> Tensor:
    r"""
    Matrix square root of a batch of symmetric, positive definite matrices.

    Computed by diagonalising the matrix :math:`X = Q_X \Lambda_X Q_X^\intercal`,
    computing the square root of the eigenvalues, and recomposing.

    :math:`\sqrt{{X}} = Q_X \sqrt{{\Lambda_X}} Q_X^\intercal`

    Note that this will be infeasible for matrices with negative eigenvalues,
    and potentially singular matrices due to numerical rounding errors. To
    guard against the infeasible case, consider specifying a ``recondition``
    parameter.

    .. note::
        This approach is in principle faster than the matrix square root in
        JAX, but it is not as robust or general as the JAX implementation
        (``jax.linalg.sqrtm``).

    :Dimension: **Input :** :math:`(N, *, D, D)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, D denotes matrix row and column
                    dimension.
                **Output :** :math:`(N, *, D, D)`
                    As above.

    Parameters
    ----------
    input : Tensor
        Batch of symmetric tensors to transform using the matrix square root.\
    {symmap_param_spec}

    Returns
    -------
    output : Tensor
        Square root of each matrix in the input batch.
    """
    return symmap(
        input,
        jnp.sqrt,
        psi=psi,
        key=key,
        recondition=recondition,
        fill_nans=fill_nans,
        truncate_eigenvalues=truncate_eigenvalues,
    )
