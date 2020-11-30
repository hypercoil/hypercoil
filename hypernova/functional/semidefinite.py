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
from hypernova import symxfm, symlog, symexp
from cov import invert_spd


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
    """
    ref_sri = symxfm(reference, torch.rsqrt, psi=recondition)
    return symlog(ref_sri @ input @ ref_sri, recondition)


def mean_euc_spd(input):
    return input.mean(0)


def mean_harm_spd(input):
    return invert_spd(invert_spd(input).mean(0))


def mean_logeuc_spd(input):
    return symexp(symlog(input).mean(0))
