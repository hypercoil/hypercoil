# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Positive semidefinite cone
~~~~~~~~~~~~~~~~~~~~~~~~~~
Initialise and compute means and mean blocks in the positive semidefinite cone.
"""
import torch
from torch.nn import Module
from functools import partial
from .base import DomainInitialiser
from ..functional import (
    mean_euc_spd, mean_harm_spd, mean_logeuc_spd, mean_geom_spd,
    SPSDNoiseSource
)


def mean_block_spd(mean_specs, data):
    """
    Apply each mean from a list of specifications to all matrices in a block.

    Dimension
    ---------
    - data : :math:`(N, *, D, D)`
        N denotes the number of observations over which each mean is computed,
        `*` denotes any number of preceding dimensions, and D denotes the size
        of each square positive semidefinite matrix. If the axis attribute of
        the mean specifications are configured appropriately, N need not
        correspond to the first axis of the input dataset.
    - output : :math:`(K, *, D, D)`
        K denotes the number of mean specs provided.

    Parameters
    ----------
    mean_specs : list(_SemidefiniteMean objects)
        List of specifications for estimating a measure of central tendency in
        the positive semidefinite cone. SemidefiniteMean subclasses are found
        at `hypercoil.init`.
    data : Tensor
        Input dataset over which each mean is to be estimated.
    """
    return torch.stack([spec(data) for spec in mean_specs]).squeeze(0)


def mean_apply_block(mean_specs, data):
    """
    Apply each mean from a list of specifications to a different slice or block
    of a dataset.

    Dimension
    ---------
    - data : :math:`(K, N, *, D, D)`
        K denotes the number of mean specs provided. N denotes the number of
        observations over which each mean is computed, `*` denotes any number
        of intervening dimensions, and D denotes the size of each square
        positive semidefinite matrix. If the axis attribute of the mean
        specifications are configured appropriately, N need not correspond to
        the first axis of the input dataset.
    - output : :math:`(K, *, D, D)`

    Parameters
    ----------
    mean_specs : list(_SemidefiniteMean objects)
        List of specifications for estimating a measure of central tendency in
        the positive semidefinite cone. SemidefiniteMean subclasses are found
        at `hypercoil.init`.
    data : Tensor
        Input dataset over which each mean is to be estimated.
    """
    return torch.stack([spec(d) for spec, d in zip(mean_specs, data)])


def tangency_init_(tensor, mean_specs, init_data, std=0):
    """
    Initialise points of tangency for projection between the positive
    semidefinite cone and a tangent subspace.

    Dimension
    ---------
    - tensor : :math:`(K, *, D, D)`
        K denotes the number of mean specs provided, D denotes the size of eac
        square positive semidefinite matrix, and `*` denotes any number of
        intervening dimensions.
    - init_data : :math:`(N, *, D, D)`
        N denotes the number of observations over which each mean is computed.
        If the axis attribute of the mean specifications are configured
        appropriately, N need not correspond to the first axis of the input
        dataset.

    Parameters
    ----------
    tensor : Tensor
        Tangency point tensor to initialise to the specified means.
    mean_specs : list(_SemidefiniteMean objects)
        List of specifications for estimating a measure of central tendency in
        the positive semidefinite cone. SemidefiniteMean subclasses are found
        at `hypercoil.init`.
    init_data : Tensor
        Input dataset over which each mean is to be estimated.
    std : float
        Standard deviation of the positive semidefinite noise added to each
        channel of the weight matrix. This can be used to ensure that different
        channels initialised from the same mean receive different gradients
        and differentiate from one another.

    Returns
    -------
        None. The tensor is initialised in-place.
    """
    means = mean_block_spd(mean_specs, init_data)
    if std > 0:
        means = SPSDNoiseSource(std=std).inject(means)
    tensor.copy_(means)


class TangencyInit(DomainInitialiser):
    def __init__(self, mean_specs, init_data, std=0, domain=None):
        if domain is not None:
            print('Warning: domain specified. If the domain mapping does not '
                  'preserve positive semidefiniteness, then the module will '
                  'likely fail on the forward pass.')
        init = partial(tangency_init_, mean_specs=mean_specs,
                       init_data=init_data, std=std)
        super(TangencyInit, self).__init__(init=init, domain=domain)


class _SemidefiniteMean(Module):
    """
    Base class for modules that compute semidefinite means.
    """
    def __init__(self, axis=0):
        super(_SemidefiniteMean, self).__init__()
        self.axis = axis

    def extra_repr(self):
        return f'axis={self.axis}'


class SPDEuclideanMean(_SemidefiniteMean):
    """
    Batch-wise Euclidean mean of tensors in the positive semidefinite cone.

    This is the familiar arithmetic mean:

    :math:`\frac{1}{N}\sum_{i=1}^N X_{i}`

    Dimension
    ---------
    - Input: :math:`(N, *, D, D)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      D denotes matrix row and column dimension. If the axis attribute is
      configured appropriately, N need not correspond to the first axis of the
      input dataset.
    - Output: :math:`(*, D, D)`

    Parameters
    ----------
    axis : int (default 0)
        Axis corresponding to observations over which the mean is computed.
    """
    def forward(self, input):
        return mean_euc_spd(input, axis=self.axis)


class SPDHarmonicMean(_SemidefiniteMean):
    """
    Batch-wise harmonic mean of tensors in the positive semidefinite cone.

    The harmonic mean is computed as the matrix inverse of the Euclidean mean
    of matrix inverses:

    :math:`\left(\frac{1}{N}\sum_{i=1}^N X_{i}^{-1}\right)^{-1}`

    Dimension
    ---------
    - Input: :math:`(N, *, D, D)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      D denotes matrix row and column dimension. If the axis attribute is
      configured appropriately, N need not correspond to the first axis of the
      input dataset.
    - Output: :math:`(*, D, D)`

    Parameters
    ----------
    axis : int (default 0)
        Axis corresponding to observations over which the mean is computed.
    """
    def forward(self, input):
        return mean_harm_spd(input, axis=self.axis)


class SPDLogEuclideanMean(_SemidefiniteMean):
    """
    Batch-wise log-Euclidean mean of tensors in the positive semidefinite cone.

    The log-Euclidean mean is computed as the matrix exponential of the mean of
    matrix logarithms.

    :math:`\exp_M \left(\frac{1}{N}\sum_{i=1}^N \log_M X_{i}\right)`

    Dimension
    ---------
    - Input: :math:`(N, *, D, D)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      D denotes matrix row and column dimension. If the axis attribute is
      configured appropriately, N need not correspond to the first axis of the
      input dataset.
    - Output: :math:`(*, D, D)`

    Parameters
    ----------
    axis : int (default 0)
        Axis corresponding to observations over which the mean is computed.
    """
    def forward(self, input):
        return mean_logeuc_spd(input, axis=self.axis)


class SPDGeometricMean(_SemidefiniteMean):
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
      D denotes matrix row and column dimension. If the axis attribute is
      configured appropriately, N need not correspond to the first axis of the
      input dataset.
    - Output: :math:`(*, D, D)`

    Parameters
    ----------
    axis : int (default 0)
        Axis corresponding to observations over which the mean is computed.
    psi : float in [0, 1]
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
    axis : int
        Axis or axes over which the mean is computed.
    """
    def __init__(self, axis=0, psi=0, eps=1e-6, max_iter=10):
        super(SPDGeometricMean, self).__init__(axis=axis)
        self.psi = psi
        self.eps = eps
        self.max_iter = max_iter

    def forward(self, input):
        return mean_geom_spd(
            input, axis=self.axis, recondition=self.psi,
            eps=self.eps, max_iter=self.max_iter
        )

    def extra_repr(self):
        s = super(SPDGeometricMean, self).extra_repr()
        if self.psi > 0:
            s += f', psi={self.psi}'
        s += f', max_iter={self.max_iter}'
        return s
