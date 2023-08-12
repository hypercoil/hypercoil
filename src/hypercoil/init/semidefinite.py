# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Initialise and compute means and mean blocks in the positive semidefinite
cone.
"""
from __future__ import annotations
from typing import Callable, Optional, Sequence, Tuple, Type, Union

import jax
import jax.numpy as jnp
import equinox as eqx
from numpyro.distributions import Normal

from hypercoil.engine.docutil import NestedDocParse
from ..engine import PyTree, Tensor
from ..engine.noise import MatrixExponential, Symmetric
from ..functional.semidefinite import (
    document_semidefinite_mean,
    mean_euc_spd,
    mean_geom_spd,
    mean_harm_spd,
    mean_logeuc_spd,
)
from .base import MappedInitialiser
from .mapparam import MappedParameter


def document_semidefinite_block_mean(f: Callable) -> Callable:
    semidefinite_block_mean_dim = """N denotes the number of observations over
                    which each mean is computed, ``*`` denotes any number of
                    preceding dimensions, and D denotes the size of each square
                    positive semidefinite matrix. If the axis attribute of the
                    mean specifications are configured appropriately, N need
                    not correspond to the first axis of the input dataset.
                **Output :** :math:`(K, *, D, D)`
                    K denotes the number of mean specs provided."""
    semidefinite_block_mean_spec = """
    mean_specs : list(_SemidefiniteMean objects)
        List of specifications for estimating a measure of central tendency in
        the positive semidefinite cone. SemidefiniteMean subclasses are found
        at `hypercoil.init`.
    data : Tensor
        Input dataset over which each mean is to be estimated."""

    fmt = NestedDocParse(
        semidefinite_block_mean_dim=semidefinite_block_mean_dim,
        semidefinite_block_mean_spec=semidefinite_block_mean_spec,
    )
    f.__doc__ = f.__doc__.format_map(fmt)
    return f


@document_semidefinite_block_mean
def mean_block_spd(
    mean_specs: Sequence[Callable],
    data: Tensor,
) -> Tensor:
    """
    Apply each mean from a list of specifications to all matrices in a block.

    :Dimension: **data :** :math:`(N, *, D, D)`
                    {semidefinite_block_mean_dim}

    Parameters
    ----------\
    {semidefinite_block_mean_spec}
    """
    return jnp.stack([spec(data) for spec in mean_specs])


@document_semidefinite_block_mean
def mean_apply_block(
    mean_specs: Sequence[Callable],
    data: Tensor,
) -> Tensor:
    """
    Apply each mean from a list of specifications to a different slice or
    block of a dataset.

    :Dimension: **data :** :math:`(K, N, *, D, D)`
                    K denotes the number of mean specs provided. \
                    {semidefinite_block_mean_dim}

    Parameters
    ----------\
    {semidefinite_block_mean_spec}
    """
    return jnp.stack([spec(d) for spec, d in zip(mean_specs, data)])


def tangency_init(
    init_data: Tensor,
    *,
    mean_specs: Sequence[Callable],
    std: float = 0.0,
    key: Optional[jax.random.PRNGKey] = None,
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
                src_distribution=Normal(0, 0.01),
                multiplicity=init_data.shape[-1],
            )
        )
        noise = src.sample(sample_shape=means.shape[:-2], key=key)
        factor = std / noise.std()
        means = means + factor * noise
    return means


class TangencyInitialiser(MappedInitialiser):
    """
    Initialise points of tangency for projection between the positive
    semidefinite cone and a tangent subspace.

    See :func:`tangency_init` for argument details.
    """

    init_data: Tensor
    mean_specs: Sequence[Callable]
    std: float

    def __init__(
        self,
        init_data: Tensor,
        mean_specs: Sequence[Callable],
        std: float = 0.0,
        mapper: Optional[Type[MappedParameter]] = None,
    ):
        self.init_data = init_data
        self.mean_specs = mean_specs
        self.std = std
        super().__init__(mapper)

    def _init(
        self,
        *,
        shape: Optional[Tuple[int, ...]] = None,
        key: jax.random.PRNGKey,
    ) -> Tensor:
        return tangency_init(
            init_data=self.init_data,
            mean_specs=self.mean_specs,
            std=self.std,
            key=key,
        )

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Optional[Type[MappedParameter]] = None,
        init_data: Tensor,
        mean_specs: Sequence[Callable],
        std: float = 0.0,
        where: Union[str, Callable] = 'weight',
        key: Optional[jax.random.PRNGKey] = None,
        **params,
    ) -> PyTree:
        init = cls(
            mapper=mapper,
            init_data=init_data,
            mean_specs=mean_specs,
            std=std,
        )
        return super()._init_impl(
            init=init, model=model, where=where, key=key, **params
        )


class _SemidefiniteMean(eqx.Module):
    """
    Base class for modules that compute semidefinite means.
    """

    axis: Union[int, Sequence[int]] = (0,)


@document_semidefinite_mean
class SPDEuclideanMean(_SemidefiniteMean):
    """\
    {euclidean_mean_desc}
    \
    {semidefinite_mean_dim}

    Parameters
    ----------\
    {semidefinite_mean_axis_spec}
    """

    def __call__(self, input: Tensor) -> Tensor:
        return mean_euc_spd(input, axis=self.axis)


@document_semidefinite_mean
class SPDHarmonicMean(_SemidefiniteMean):
    """\
    {harmonic_mean_desc}
    \
    {semidefinite_mean_dim}

    Parameters
    ----------\
    {semidefinite_mean_axis_spec}
    """

    def __call__(self, input: Tensor) -> Tensor:
        return mean_harm_spd(input, axis=self.axis)


@document_semidefinite_mean
class SPDLogEuclideanMean(_SemidefiniteMean):
    """\
    {logeuc_mean_desc}
    \
    {semidefinite_mean_dim}

    Parameters
    ----------\
    {semidefinite_mean_axis_spec}\
    {semidefinite_psi_spec}
    """

    psi: float = 0.0

    def __call__(self, input: Tensor) -> Tensor:
        return mean_logeuc_spd(
            input,
            axis=self.axis,
            psi=self.psi,
            recondition='convexcombination',
        )


@document_semidefinite_mean
class SPDGeometricMean(_SemidefiniteMean):
    r"""\
    {geometric_mean_desc}
    \
    {semidefinite_mean_dim}

    Parameters
    ----------\
    {semidefinite_mean_axis_spec}\
    {semidefinite_mean_psi_spec}\
    {semidefinite_mean_maxiter_spec}
    """
    psi: float = 0.0
    eps: float = 1e-5
    max_iter: int = 10

    def __call__(self, input: Tensor) -> Tensor:
        return mean_geom_spd(
            input,
            axis=self.axis,
            psi=self.psi,
            max_iter=self.max_iter,
            recondition='convexcombination',
        )
