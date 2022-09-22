# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Initialise and compute means and mean blocks in the positive semidefinite
cone.
"""
import jax
import jax.numpy as jnp
import distrax
import equinox as eqx
from typing import Callable, Optional, Sequence, Tuple, Type, Union
from .base import MappedInitialiser
from .mapparam import MappedParameter
from ..engine import PyTree, Tensor
from ..engine.noise import (
    Symmetric, MatrixExponential
)
from ..functional import (
    mean_euc_spd, mean_harm_spd,
    mean_logeuc_spd, mean_geom_spd,
)


def mean_block_spd(
    mean_specs: list[Callable],
    data: Tensor
) -> Tensor:
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
    return jnp.stack([spec(data) for spec in mean_specs])


def mean_apply_block(
    mean_specs: list[Callable],
    data: Tensor
) -> Tensor:
    """
    Apply each mean from a list of specifications to a different slice or
    block of a dataset.

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
    return jnp.stack([spec(d) for spec, d in zip(mean_specs, data)])


def tangency_init(
    init_data: Tensor,
    *,
    mean_specs: list[Callable],
    std: float = 0.,
    key: Optional[jax.random.PRNGKey] = None
) -> Tensor:
    """
    Initialise points of tangency for projection between the positive
    semidefinite cone and a tangent subspace.

    :Dimension: **init_data :** :math:`(N, *, D, D)` or :math:`(N, *, obs, C)`
                    N denotes the number of observations over which each mean
                    is computed. If the axis attribute of the mean
                    specifications are configured appropriately, N need not
                    correspond to the first axis of the input dataset.

    Parameters
    ----------
    init_data : Tensor
        Input dataset over which each mean is to be estimated.
    mean_specs : list(``_SemidefiniteMean`` objects)
        List of specifications for estimating a measure of central tendency in
        the positive semidefinite cone. ``_SemidefiniteMean`` subclasses are
        found at
        :doc:`hypercoil.init.semidefinite <hypercoil.init.semidefinite.SemidefiniteMean>`.
    std : float
        Standard deviation of the positive semidefinite noise added to each
        channel of the weight matrix. This can be used to ensure that
        different channels initialised from the same mean receive different
        gradients and differentiate from one another.

    Returns
    -------
    Tensor
        The initialised tensor.
    """
    means = mean_block_spd(mean_specs, init_data)
    if std > 0:
        src = MatrixExponential(
            Symmetric(
                src_distribution=distrax.Normal(0, 0.01),
                multiplicity=init_data.shape[-1]
            )
        )
        noise = src.sample(sample_shape=means.shape[:-2], seed=key)
        factor = std / noise.std()
        means = means + factor * noise
    return means


class TangencyInitialiser(MappedInitialiser):
    """
    Initialise points of tangency for projection between the positive
    semidefinite cone and a tangent subspace.

    See :func:`tangency_init` for argument details.
    """

    init_data : Tensor
    mean_specs : Sequence[Callable]
    std : float

    def __init__(
        self,
        init_data: Tensor,
        mean_specs: Sequence[Callable],
        std: float = 0.,
        mapper: Optional[Type[MappedParameter]] = None
    ):
        self.init_data = init_data
        self.mean_specs = mean_specs
        self.std = std
        super().__init__(mapper)

    def _init(
        self,
        *,
        shape: Optional[Tuple[int, ...]] = None,
        key: jax.random.PRNGKey
    ) -> Tensor:
        return tangency_init(
            init_data=self.init_data,
            mean_specs=self.mean_specs,
            std=self.std,
            key=key
        )

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Optional[Type[MappedParameter]] = None,
        init_data: Tensor,
        mean_specs: Sequence[Callable],
        std: float = 0.,
        param_name: str = "weight",
        key: Optional[jax.random.PRNGKey] = None,
        **params,
    ) -> PyTree:
        init = cls(
            mapper=mapper,
            init_data=init_data,
            mean_specs=mean_specs,
            std=std
        )
        return super()._init_impl(
            init=init, model=model, param_name=param_name, key=key, **params)


class _SemidefiniteMean(eqx.Module):
    """
    Base class for modules that compute semidefinite means.
    """
    axis: Union[int, Sequence[int]] = (0,)


class SPDEuclideanMean(_SemidefiniteMean):
    r"""
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
    def __call__(self, input):
        return mean_euc_spd(input, axis=self.axis)


class SPDHarmonicMean(_SemidefiniteMean):
    r"""
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
    def __call__(self, input):
        return mean_harm_spd(input, axis=self.axis)


class SPDLogEuclideanMean(_SemidefiniteMean):
    r"""
    Batch-wise log-Euclidean mean of tensors in the positive semidefinite
    cone.

    The log-Euclidean mean is computed as the matrix exponential of the mean
    of matrix logarithms.

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
    psi: float = 0.

    def __call__(self, input):
        return mean_logeuc_spd(input, axis=self.axis, psi=self.psi,
                               recondition='convexcombination')


class SPDGeometricMean(_SemidefiniteMean):
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
       the same point of tangency. It now becomes a new working estimate of
       the mean and thus a new point of tangency.
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
    """
    psi: float = 0.
    eps: float = 1e-5
    max_iter: int = 10

    def __call__(self, input):
        return mean_geom_spd(
            input, axis=self.axis, psi=self.psi,
            eps=self.eps, max_iter=self.max_iter,
            recondition='convexcombination'
        )
