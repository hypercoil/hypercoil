# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Differentiable projection from the positive semidefinite cone into a proper
subspace tangent to the Riemann manifold.

.. warning::
    Nearly all operations here, as currently implemented, exhibit numerical
    instability in both forward and backward passes.
"""
import jax.numpy as jnp
from typing import Literal, Optional, Sequence, Union
from . import symmap, symlog, symexp, symsqrt, spd
from .symmap import document_symmetric_map
from ..engine import Tensor


@document_symmetric_map
def tangent_project_spd(
    input: Tensor,
    reference: Tensor,
    psi: float = 0,
    key: Optional[Tensor] = None,
    recondition: Literal['eigenspaces', 'convexcombination'] = 'eigenspaces',
    fill_nans: bool = True,
    truncate_eigenvalues: bool = False,
) -> Tensor:
    r"""
    Project a batch of symmetric matrices from the positive semidefinite cone
    into a tangent subspace.

    Given a tangency point :math:`\Omega`, each input :math:`\Theta` is
    projected as:

    :math:`\vec{{\Theta}} = \log \Omega^{{-1/2}} \Theta \Omega^{{-1/2}}`

    where :math:`\Omega^{{-1/2}}` denotes the inverse matrix square root of
    :math:`\Omega` and :math:`\log` denotes the matrix-argument logarithm.

    :Dimension: **Input :** :math:`(N, *, D, D)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, D denotes matrix row and column
                    dimension.
                **Reference :** :math:`(*, D, D)`
                    As above.
                **Output :** :math:`(N, *, D, D)`
                    As above.

    Parameters
    ----------
    input : Tensor
        Batch of symmetric positive definite matrices to project into the
        tangent subspace.
    reference : Tensor
        Point of tangency. This is an element of the positive semidefinite
        cone at which the projection occurs. It should be representative of
        the sample being projected (for instance, some form of mean).
    {param_spec}

    Returns
    -------
    output : Tensor
        Batch of matrices transformed via projection into the tangent subspace.

    See also
    --------
    cone_project_spd: The inverse projection, into the semidefinite cone.
    """
    ref_sri = symmap(
        reference, lambda x: x ** -0.5,
        psi=psi, key=key, recondition=recondition,
        fill_nans=fill_nans, truncate_eigenvalues=truncate_eigenvalues)
    return symlog(
        ref_sri @ input @ ref_sri,
        psi=psi, key=key, recondition=recondition,
        fill_nans=fill_nans, truncate_eigenvalues=truncate_eigenvalues)


@document_symmetric_map
def cone_project_spd(
    input: Tensor,
    reference: Tensor,
    psi: float = 0,
    key: Optional[Tensor] = None,
    recondition: Literal['eigenspaces', 'convexcombination'] = 'eigenspaces',
    fill_nans: bool = True,
    truncate_eigenvalues: bool = False,
) -> Tensor:
    r"""
    Project a batch of symmetric matrices from a tangent subspace into the
    positive semidefinite cone.

    Given a tangency point :math:`\Omega`, each input :math:`\vec{{\Theta}}` is
    projected as:

    :math:`\Theta = \Omega^{{1/2}} \exp \vec{{\Theta}} \Omega^{{1/2}}`

    where :math:`\Omega^{{1/2}}` denotes the matrix square root of :math:`\Omega`
    and :math:`\exp` denotes the matrix-argument exponential.

    .. warning::
        If the tangency point is not in the positive semidefinite cone, the
        result is undefined unless reconditioning is used. If reconditioning
        is necessary, however, ``cone_project_spd`` is not guaranteed to be
        a well-formed inverse of ``tangent_project_spd``.

    :Dimension: **Input :** :math:`(N, *, D, D)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, D denotes matrix row and column
                    dimension.
                **Reference :** :math:`(*, D, D)`
                    As above.
                **Output :** :math:`(N, *, D, D)`
                    As above.

    Parameters
    ----------
    input : Tensor
        Batch of matrices to project into the positive semidefinite cone.
    reference : Tensor
        Point of tangency. This is an element of the positive semidefinite
        cone at which the projection occurs. It should be representative of
        the sample being projected (for instance, some form of mean).
    {param_spec}

    Returns
    -------
    output : Tensor
        Batch of matrices transformed via projection into the positive
        semidefinite cone.

    See also
    --------
    tangent_project_spd: The inverse projection, into a tangent subspace.
    """
    ref_sr = symsqrt(
        reference,
        psi=psi, key=key, recondition=recondition,
        fill_nans=fill_nans, truncate_eigenvalues=truncate_eigenvalues)
    cone = ref_sr @ symexp(input) @ ref_sr
    # Note that we do not undo the reconditioning, and so the map is not
    # a well-formed inverse of the tangent_project_spd map if the input is not
    # positive definite to begin with.
    return spd(cone)


def mean_euc_spd(
    input: Tensor,
    axis: Union[int, Sequence[int]] = 0,
) -> Tensor:
    r"""
    Batch-wise Euclidean mean of tensors in the positive semidefinite cone.

    This is the familiar arithmetic mean:

    :math:`\bar{X} = \frac{1}{N}\sum_{i=1}^N X_{i}`

    :Dimension: **Input :** :math:`(N, *, D, D)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, D denotes matrix row and column
                    dimension.
                **Output :** :math:`(*, D, D)`
                    As above.

    Dimension
    ---------
    - Input: :math:`(N, *, D, D)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      D denotes matrix row and column dimension
    - Output: :math:`(*, D, D)`

    Parameters
    ----------
    input : Tensor
        Batch of matrices over which the Euclidean mean is to be computed.
    axis : int
        Axis or axes over which the mean is computed.

    Returns
    -------
    output : Tensor
        Euclidean mean of the input batch.
    """
    return input.mean(axis)


def mean_harm_spd(
    input: Tensor,
    axis: Union[int, Sequence[int]] = 0,
    require_nonsingular: bool = True
) -> Tensor:
    r"""
    Batch-wise harmonic mean of tensors in the positive semidefinite cone.

    The harmonic mean is computed as the matrix inverse of the Euclidean mean
    of matrix inverses:

    :math:`\bar{X} = \left(\frac{1}{N}\sum_{i=1}^N X_{i}^{-1}\right)^{-1}`

    :Dimension: **Input :** :math:`(N, *, D, D)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, D denotes matrix row and column
                    dimension.
                **Output :** :math:`(*, D, D)`
                    As above.

    Parameters
    ----------
    input : Tensor
        Batch of matrices over which the harmonic mean is to be computed.
    axis : int (default 0)
        Axis or axes over which the mean is computed.
    require_nonsingular : bool (default True)
        Indicates that the input matrix must be nonsingular. If this is
        False, then the Moore-Penrose pseudoinverse is computed instead of
        the inverse.

    Returns
    -------
    output : Tensor
        Harmonic mean of the input batch.
    """
    inverse = jnp.linalg.inv if require_nonsingular else jnp.linalg.pinv
    return inverse(inverse(input).mean(axis))


@document_symmetric_map
def mean_logeuc_spd(
    input: Tensor,
    axis: Union[int, Sequence[int]] = 0,
    psi: float = 0,
    key: Optional[Tensor] = None,
    recondition: Literal['eigenspaces', 'convexcombination'] = 'eigenspaces',
    fill_nans: bool = True,
    truncate_eigenvalues: bool = False,
) -> Tensor:
    r"""
    Batch-wise log-Euclidean mean of tensors in the positive semidefinite cone.

    The log-Euclidean mean is computed as the matrix exponential of the mean of
    matrix logarithms.

    :math:`\bar{{X}} = \exp \left(\frac{{1}}{{N}}\sum_{{i=1}}^N \log X_{{i}}\right)`

    :Dimension: **Input :** :math:`(N, *, D, D)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, D denotes matrix row and column
                    dimension.
                **Output :** :math:`(*, D, D)`
                    As above.

    Parameters
    ----------
    input : Tensor
        Batch of matrices over which the log-Euclidean mean is to be computed.
    axis : int
        Axis or axes over which the mean is computed.
    {param_spec}

    Returns
    -------
    output : Tensor
        Log-Euclidean mean of the input batch.
    """
    return symexp(symlog(
        input, psi=psi, key=key, recondition=recondition,
        fill_nans=fill_nans, truncate_eigenvalues=truncate_eigenvalues
    ).mean(axis))


#TODO: Reformulate this as an optimiser / descent algorithm. Implement
#      the gradient using the implicit function theorem if possible.
@document_symmetric_map
def mean_geom_spd(
    input: Tensor,
    axis: Union[int, Sequence[int]] = 0,
    eps: float = 1e-6,
    max_iter: int = 10,
    psi: float = 0,
    key: Optional[Tensor] = None,
    recondition: Literal['eigenspaces', 'convexcombination'] = 'eigenspaces',
    fill_nans: bool = True,
    truncate_eigenvalues: bool = False,
) -> Tensor:
    r"""
    Batch-wise geometric mean of tensors in the positive semidefinite cone.

    The geometric mean is computed via gradient descent along the geodesic on
    the manifold. In brief:

    Initialisation :
     - The estimate of the mean is initialised to the Euclidean mean.
    Iteration :
     - Using the working estimate of the mean as the point of tangency, the
       tensors are projected into a tangent space.
     - The arithmetic mean of the tensors is computed in tangent space.
     - This mean is projected back into the positive semidefinite cone using
       the same point of tangency. It now becomes a new working estimate of the
       mean and thus a new point of tangency.
    Termination / convergence :
     - The algorithm terminates either when the Frobenius norm of the
       difference between the new estimate and the previous estimate is less
       than a specified threshold, or when a maximum number of iterations has
       been attained.

    :Dimension: **Input :** :math:`(N, *, D, D)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, D denotes matrix row and column
                    dimension.
                **Output :** :math:`(*, D, D)`
                    As above.

    Parameters
    ----------
    input : Tensor
        Batch of matrices over which the geometric mean is to be computed.
    axis : int
        Axis or axes over which the mean is computed.
    eps : float
        The minimum value of the Frobenius norm required for convergence.
    max_iter : nonnegative int
        The maximum number of iterations of gradient descent to run before
        termination.
    {param_spec}

    Returns
    -------
    output : Tensor
        Geometric mean of the input batch.
    """
    ref = mean_euc_spd(input, axis)
    for _ in range(max_iter):
        tan = tangent_project_spd(
            input, ref, psi=psi, key=key, recondition=recondition,
            fill_nans=fill_nans, truncate_eigenvalues=truncate_eigenvalues)
        reftan = tan.mean(axis)
        ref_old = ref
        ref = cone_project_spd(
            reftan, ref, psi=psi, key=key, recondition=recondition,
            fill_nans=fill_nans, truncate_eigenvalues=truncate_eigenvalues)
        if jnp.all(jnp.linalg.norm(ref - ref_old, ord='fro', axis=(-1, -2)) < eps):
            break
    return ref


#TODO: marking this as an experimental function
def mean_kullback_spd(input, alpha, recondition=0):
    S = symsqrt(mean_euc_spd(input), recondition)
    R = symmap(mean_euc_spd(input), lambda X: X ** -0.5, psi=recondition)
    T = R @ mean_harm_spd(input) @ R
    return S @ symmap(T, lambda X: jnp.power(X, alpha)) @ S
