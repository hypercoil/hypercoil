# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Positive semidefinite cone
~~~~~~~~~~~~~~~~~~~~~~~~~~
Differentiable projection from the positive semidefinite cone into a proper
subspace tangent to the Riemann manifold.
"""
import torch
from . import symmap, symlog, symexp, symsqrt, invert_spd


def tangent_project_spd(input, reference, recondition=0):
    """
    Project a batch of symmetric matrices from the positive semidefinite cone
    into a tangent subspace.

    Given a tangency point :math:`\Omega`, each input :math:`\Theta` is
    projected as:

    :math:`\bar{Theta} = \log_M \Omega^{-1/2} \Theta \Omega^{-1/2}`

    where :math:`\Omega^{-1/2}` denotes the inverse matrix square root of
    :math:`\Omega` and :math:`log_M` denotes the matrix-valued logarithm.

    Dimension
    ---------
    - Input: :math:`(N, *, D, D)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      D denotes matrix row and column dimension
    - Reference: :math:`(*, D, D)`
    - Output: :math:`(N, *, D, D)`

    Parameters
    ----------
    input : Tensor
        Batch of symmetric positive definite matrices to project into the
        tangent subspace.
    reference : Tensor
        Point of tangency. This is an element of the positive semidefinite
        cone at which the projection occurs. It should be representative of
        the sample being projected (for instance, some form of mean).
    recondition : float in [0, 1]
        Conditioning factor to promote positive definiteness. If this is in
        (0, 1], the original input will be replaced with a convex combination
        of the input and an identity matrix.

        :math:`\hat{X} = (1 - \psi) X + \psi I`

        A suitable value can be used to ensure that all eigenvalues are
        positive and therefore guarantee that the matrix is in the domain.

    Returns
    -------
    output : Tensor
        Batch of matrices transformed via projection into the tangent subspace.

    See also
    --------
    cone_project_spd: The inverse projection, into the semidefinite cone.
    """
    ref_sri = symmap(reference, torch.rsqrt, psi=recondition)
    return symlog(ref_sri @ input @ ref_sri, recondition)


def cone_project_spd(input, reference, recondition=0):
    """
    Project a batch of symmetric matrices from a tangent subspace into the
    positive semidefinite cone.

    Given a tangency point :math:`\Omega`, each input :math:`\Theta` is
    projected as:

    :math:`\bar{Theta} = \Omega^{1/2} \exp_M \Theta \Omega^{1/2}`

    where :math:`\Omega^{1/2}` denotes the matrix square root of :math:`\Omega`
    and :math:`exp_M` denotes the matrix-valued exponential.

    Dimension
    ---------
    - Input: :math:`(N, *, D, D)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      D denotes matrix row and column dimension
    - Reference: :math:`(*, D, D)`
    - Output: :math:`(N, *, D, D)`

    Parameters
    ----------
    input : Tensor
        Batch of matrices to project into the positive semidefinite cone.
    reference : Tensor
        Point of tangency. This is an element of the positive semidefinite
        cone at which the projection occurs. It should be representative of
        the sample being projected (for instance, some form of mean).
    recondition : float in [0, 1]
        Conditioning factor to promote positive definiteness. If this is in
        (0, 1], the original input will be replaced with a convex combination
        of the input and an identity matrix.

        :math:`\hat{X} = (1 - \psi) X + \psi I`

        A suitable value can be used to ensure that all eigenvalues are
        positive and therefore guarantee that the matrix is in the domain.

    Returns
    -------
    output : Tensor
        Batch of matrices transformed via projection into the positive
        semidefinite cone.

    See also
    --------
    tangent_project_spd: The inverse projection, into a tangent subspace.
    """
    ref_sr = symsqrt(reference, recondition)
    # Note that we recondition the exponential as well to ensure that this is
    # a well-formed inverse of `tangent_project_spd`.
    if recondition > 0:
        recondition * input + (1 - recondition) * torch.eye(input.size(-1))
    # Note that we must use the much slower torch.matrix_exp for stability.
    return ref_sr @ torch.matrix_exp(input) @ ref_sr


def mean_euc_spd(input):
    """
    Batch-wise Euclidean mean of tensors in the positive semidefinite cone.

    This is the familiar arithmetic mean:

    :math:`\frac{1}{N}\sum_{i=1}^N X_{i}`

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

    Returns
    -------
    output : Tensor
        Euclidean mean of the input batch.
    """
    return input.mean(0)


def mean_harm_spd(input):
    """
    Batch-wise harmonic mean of tensors in the positive semidefinite cone.

    The harmonic mean is computed as the matrix inverse of the Euclidean mean
    of matrix inverses:

    :math:`\left(\frac{1}{N}\sum_{i=1}^N X_{i}^{-1}\right)^{-1}`

    Dimension
    ---------
    - Input: :math:`(N, *, D, D)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      D denotes matrix row and column dimension
    - Output: :math:`(*, D, D)`

    Parameters
    ----------
    input : Tensor
        Batch of matrices over which the harmonic mean is to be computed.

    Returns
    -------
    output : Tensor
        Harmonic mean of the input batch.
    """
    return invert_spd(invert_spd(input).mean(0))


def mean_logeuc_spd(input):
    """
    Batch-wise log-Euclidean mean of tensors in the positive semidefinite cone.

    The log-Euclidean mean is computed as the matrix exponential of the mean of
    matrix logarithms.

    :math:`\exp_M \left(\frac{1}{N}\sum_{i=1}^N \log_M X_{i}\right)`

    Dimension
    ---------
    - Input: :math:`(N, *, D, D)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      D denotes matrix row and column dimension
    - Output: :math:`(*, D, D)`

    Parameters
    ----------
    input : Tensor
        Batch of matrices over which the log-Euclidean mean is to be computed.

    Returns
    -------
    output : Tensor
        Log-Euclidean mean of the input batch.
    """
    return symexp(symlog(input).mean(0))


def mean_geom_spd(input, recondition=0, eps=1e-6, max_iter=10):
    """
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

    Dimension
    ---------
    - Input: :math:`(N, *, D, D)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      D denotes matrix row and column dimension
    - Output: :math:`(*, D, D)`

    Parameters
    ----------
    input : Tensor
        Batch of matrices over which the geometric mean is to be computed.
    recondition : float in [0, 1]
        Conditioning factor to promote positive definiteness. If this is in
        (0, 1], the original input will be replaced with a convex combination
        of the input and an identity matrix.

        :math:`\hat{X} = (1 - \psi) X + \psi I`

        A suitable value can be used to ensure that all eigenvalues are
        positive and therefore guarantee that the matrix is in the domain of
        projection operations.
    eps : float
        The minimum value of the Frobenius norm required for convergence.
    max_iter : nonnegative int
        The maximum number of iterations of gradient descent to run before
        termination.

    Returns
    -------
    output : Tensor
        Geometric mean of the input batch.
    """
    ref = mean_euc_spd(input)
    for i in range(max_iter):
        tan = tangent_project_spd(input, ref, recondition)
        reftan = tan.mean(0)
        ref_old = ref
        ref = cone_project_spd(reftan, ref, recondition)
        if torch.all(torch.norm(ref, dim=(-1, -2)) < eps):
            break
    return ref


def mean_kullback_spd(input, alpha, recondition=0):
    S = symsqrt(mean_euc_spd(input), recondition)
    R = symmap(mean_euc_spd(input), torch.rsqrt, psi=recondition)
    T = R @ mean_harm_spd(input) @ R
    return S @ symmap(T, lambda X: torch.pow(X, alpha)) @ S
