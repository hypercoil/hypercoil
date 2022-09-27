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
from typing import Callable, Literal, Optional, Sequence, Union
from hypercoil.functional.matrix import spd
from hypercoil.functional.symmap import (
    symmap, symlog, symexp, symsqrt, document_symmetric_map
)
from ..engine import NestedDocParse, Tensor


def document_semidefinite_projection(f: Callable) -> Callable:
    tangent_project_long_desc = r"""
    Given a tangency point :math:`\Omega`, each input :math:`\Theta` is
    projected as:

    :math:`\vec{{\Theta}} = \log \Omega^{{-1/2}} \Theta \Omega^{{-1/2}}`

    where :math:`\Omega^{{-1/2}}` denotes the inverse matrix square root of
    :math:`\Omega` and :math:`\log` denotes the matrix-argument logarithm."""
    cone_project_long_desc = r"""
    Given a tangency point :math:`\Omega`, each input :math:`\vec{{\Theta}}` is
    projected as:

    :math:`\Theta = \Omega^{{1/2}} \exp \vec{{\Theta}} \Omega^{{1/2}}`

    where :math:`\Omega^{{1/2}}` denotes the matrix square root of :math:`\Omega`
    and :math:`\exp` denotes the matrix-argument exponential.

    .. warning::
        If the tangency point is not in the positive semidefinite cone, the
        result is undefined unless reconditioning is used. If reconditioning
        is necessary, however, ``cone_project_spd`` is not guaranteed to be
        a well-formed inverse of ``tangent_project_spd``."""
    tangency_project_dim = """
    :Dimension: **Input :** :math:`(N, *, D, D)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, D denotes matrix row and column
                    dimension.
                **Reference :** :math:`(*, D, D)`
                    As above.
                **Output :** :math:`(N, *, D, D)`
                    As above."""
    tangency_base_spec = """
    input : Tensor
        Batch of symmetric positive definite matrices to project into the
        tangent subspace.
    reference : Tensor
        Point of tangency. This is an element of the positive semidefinite
        cone at which the projection occurs. It should be representative of
        the sample being projected (for instance, some form of mean)."""

    fmt = NestedDocParse(
        tangency_base_spec=tangency_base_spec,
        tangency_project_dim=tangency_project_dim,
        tangent_project_long_desc=tangent_project_long_desc,
        cone_project_long_desc=cone_project_long_desc,
    )
    f.__doc__ = f.__doc__.format_map(fmt)
    return f


def document_semidefinite_mean(f: Callable) -> Callable:
    euclidean_mean_desc = r"""
    Batch-wise Euclidean mean of tensors in the positive semidefinite cone.

    This is the familiar arithmetic mean:

    :math:`\frac{1}{N}\sum_{i=1}^N X_{i}`"""

    harmonic_mean_desc = r"""
    The harmonic mean is computed as the matrix inverse of the Euclidean mean
    of matrix inverses:

    :math:`\bar{X} = \left(\frac{1}{N}\sum_{i=1}^N X_{i}^{-1}\right)^{-1}`"""

    logeuc_mean_desc = r"""
    Batch-wise log-Euclidean mean of tensors in the positive semidefinite cone.

    The log-Euclidean mean is computed as the matrix exponential of the mean of
    matrix logarithms.

    :math:`\bar{{X}} = \exp \left(\frac{{1}}{{N}}\sum_{{i=1}}^N \log X_{{i}}\right)`"""

    semidefinite_mean_dim = r"""
    :Dimension: **Input :** :math:`(N, *, D, D)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, D denotes matrix row and column
                    dimension.
                **Output :** :math:`(*, D, D)`
                    As above."""

    geometric_mean_desc = r"""
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
       been attained."""

    semidefinite_mean_input_spec = """
    input : Tensor
        Batch of matrices over which the Euclidean mean is to be computed."""
    semidefinite_mean_axis_spec = """
    axis : int
        Axis or axes over which the mean is computed."""
    semidefinite_mean_nonsingular_spec = """
    require_nonsingular : bool (default True)
        Indicates that the input matrix must be nonsingular. If this is
        False, then the Moore-Penrose pseudoinverse is computed instead of
        the inverse."""
    semidefinite_mean_psi_spec = r"""
    psi : float in [0, 1]
        Conditioning factor to promote positive definiteness. If this is in
        (0, 1], the original input will be replaced with a convex combination
        of the input and an identity matrix.

        :math:`\hat{X} = (1 - \psi) X + \psi I`

        A suitable value can be used to ensure that all eigenvalues are
        positive and therefore guarantee that the matrix is in the domain of
        projection operations."""
    semidefinite_mean_maxiter_spec = """
    max_iter : nonnegative int
        The maximum number of iterations of gradient descent to run before
        termination."""
    semidefinite_mean_return_spec = """
    Returns
    -------
    output : Tensor
        Log-Euclidean mean of the input batch."""

    fmt = NestedDocParse(
        euclidean_mean_desc=euclidean_mean_desc,
        harmonic_mean_desc=harmonic_mean_desc,
        logeuc_mean_desc=logeuc_mean_desc,
        geometric_mean_desc=geometric_mean_desc,
        semidefinite_mean_dim=semidefinite_mean_dim,
        semidefinite_mean_input_spec=semidefinite_mean_input_spec,
        semidefinite_mean_axis_spec=semidefinite_mean_axis_spec,
        semidefinite_mean_nonsingular_spec=semidefinite_mean_nonsingular_spec,
        semidefinite_mean_psi_spec=semidefinite_mean_psi_spec,
        semidefinite_mean_maxiter_spec=semidefinite_mean_maxiter_spec,
        semidefinite_mean_return_spec=semidefinite_mean_return_spec,
    )
    f.__doc__ = f.__doc__.format_map(fmt)
    return f


@document_symmetric_map
@document_semidefinite_projection
def tangent_project_spd(
    input: Tensor,
    reference: Tensor,
    psi: float = 0,
    key: Optional[Tensor] = None,
    recondition: Literal['eigenspaces', 'convexcombination'] = 'eigenspaces',
    fill_nans: bool = True,
    truncate_eigenvalues: bool = False,
) -> Tensor:
    """
    Project a batch of symmetric matrices from the positive semidefinite cone
    into a tangent subspace.
    \
    {tangent_project_long_desc}
    \
    {tangency_project_dim}

    Parameters
    ----------\
    {tangency_base_spec}\
    {symmap_param_spec}

    Returns
    -------
    Tensor
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
@document_semidefinite_projection
def cone_project_spd(
    input: Tensor,
    reference: Tensor,
    psi: float = 0,
    key: Optional[Tensor] = None,
    recondition: Literal['eigenspaces', 'convexcombination'] = 'eigenspaces',
    fill_nans: bool = True,
    truncate_eigenvalues: bool = False,
) -> Tensor:
    """
    Project a batch of symmetric matrices from a tangent subspace into the
    positive semidefinite cone.
    \
    {cone_project_long_desc}
    \
    {tangency_project_dim}

    Parameters
    ----------\
    {tangency_base_spec}\
    {symmap_param_spec}

    Returns
    -------
    Tensor
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


@document_semidefinite_mean
def mean_euc_spd(
    input: Tensor,
    axis: Union[int, Sequence[int]] = 0,
) -> Tensor:
    """\
    {euclidean_mean_desc}
    \
    {semidefinite_mean_dim}

    Parameters
    ----------\
    {semidefinite_mean_input_spec}\
    {semidefinite_mean_axis_spec}
    \
    {semidefinite_mean_return_spec}
    """
    return input.mean(axis)


@document_semidefinite_mean
def mean_harm_spd(
    input: Tensor,
    axis: Union[int, Sequence[int]] = 0,
    require_nonsingular: bool = True
) -> Tensor:
    """\
    {harmonic_mean_desc}
    \
    {semidefinite_mean_dim}

    Parameters
    ----------\
    {semidefinite_mean_input_spec}\
    {semidefinite_mean_axis_spec}\
    {semidefinite_mean_nonsingular_spec}
    \
    {semidefinite_mean_return_spec}
    """
    inverse = jnp.linalg.inv if require_nonsingular else jnp.linalg.pinv
    return inverse(inverse(input).mean(axis))


@document_symmetric_map
@document_semidefinite_mean
def mean_logeuc_spd(
    input: Tensor,
    axis: Union[int, Sequence[int]] = 0,
    psi: float = 0,
    key: Optional[Tensor] = None,
    recondition: Literal['eigenspaces', 'convexcombination'] = 'eigenspaces',
    fill_nans: bool = True,
    truncate_eigenvalues: bool = False,
) -> Tensor:
    """\
    {logeuc_mean_desc}
    \
    {semidefinite_mean_dim}

    Parameters
    ----------\
    {semidefinite_mean_input_spec}\
    {semidefinite_mean_axis_spec}\
    {symmap_param_spec}\
    {semidefinite_mean_nonsingular_spec}
    \
    {semidefinite_mean_return_spec}
    """
    return symexp(symlog(
        input, psi=psi, key=key, recondition=recondition,
        fill_nans=fill_nans, truncate_eigenvalues=truncate_eigenvalues
    ).mean(axis))


#TODO: Reformulate this as an optimiser / descent algorithm. Implement
#      the gradient using the implicit function theorem if possible.
@document_symmetric_map
@document_semidefinite_mean
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
    """\
    {geometric_mean_desc}
    \
    {semidefinite_mean_dim}

    Parameters
    ----------\
    {semidefinite_mean_input_spec}\
    {semidefinite_mean_axis_spec}\
    {semidefinite_mean_maxiter_spec}\
    {symmap_param_spec}
    \
    {semidefinite_mean_return_spec}
    """
    ref = mean_euc_spd(input, axis)
    for _ in range(max_iter):
        tan = tangent_project_spd(
            input, ref, psi=psi, key=key, recondition=recondition,
            fill_nans=fill_nans, truncate_eigenvalues=truncate_eigenvalues)
        reftan = tan.mean(axis)
        ref = cone_project_spd(
            reftan, ref, psi=psi, key=key, recondition=recondition,
            fill_nans=fill_nans, truncate_eigenvalues=truncate_eigenvalues)
    return ref


#TODO: marking this as an experimental function
def mean_kullback_spd(input, alpha, recondition=0):
    S = symsqrt(mean_euc_spd(input), recondition)
    R = symmap(mean_euc_spd(input), lambda X: X ** -0.5, psi=recondition)
    T = R @ mean_harm_spd(input) @ R
    return S @ symmap(T, lambda X: jnp.power(X, alpha)) @ S
