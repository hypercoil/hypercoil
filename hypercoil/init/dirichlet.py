# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Initialise a tensor such that elements along a given axis are Dirichlet
samples.
"""
from __future__ import annotations
from typing import Callable, Optional, Sequence, Tuple, Type, Union

import jax
import jax.numpy as jnp
from numpyro.distributions import Dirichlet, Distribution

from ..engine import PyTree, Tensor
from ..engine.noise import sample_multivariate
from .base import MappedInitialiser
from .mapparam import MappedParameter, ProbabilitySimplexParameter


def dirichlet_init(
    *,
    shape: Tuple[int],
    distr: Distribution,
    axis: int = -1,
    key: jax.random.PRNGKey,
) -> Tensor:
    """
    Dirichlet sample initialisation.

    Initialise a tensor such that any 1D slice through that tensor along a
    given axis is a sample from a specified Dirichlet distribution. Each 1D
    slice can therefore be understood as encoding a categorical probability
    distribution.

    Parameters
    ----------
    shape : tuple
        Shape of the tensor to initialise.
    distr : instance of ``torch.distributions.Dirichlet``
        Parametrised Dirichlet distribution from which all 1D slices of the
        input tensor along the specified axis are sampled.
    axis : int (default -1)
        Axis along which slices are sampled from the specified Dirichlet
        distribution.
    key : jax.random.PRNGKey
        Pseudo-random number generator key for sampling the Dirichlet
        distribution.
    """
    return sample_multivariate(
        distr=distr, shape=shape, event_axes=(axis,), key=key
    )


class DirichletInitialiser(MappedInitialiser):
    """
    Initialise a parameter such that all slices along the final axis are
    samples from a specified Dirichlet distribution.

    See :func:`dirichlet_init` and :class:`MappedInitialiser` for argument
    details.
    """

    distr: Distribution
    axis: int = -1

    def __init__(
        self,
        concentration: Sequence[float],
        num_classes: Optional[int] = None,
        axis: int = -1,
        mapper: Optional[Type[MappedParameter]] = None,
    ):
        if len(concentration) == 1:
            concentration = concentration * num_classes
        else:
            concentration = concentration
        concentration = jnp.asarray(concentration)
        self.distr = Dirichlet(concentration=concentration)
        self.axis = axis
        super().__init__(mapper=mapper)

    def _init(
        self,
        shape: Tuple[int, ...],
        key: jax.random.PRNGKey,
    ) -> Tensor:
        return dirichlet_init(
            shape=shape, distr=self.distr, axis=self.axis, key=key
        )

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Optional[Type[MappedParameter]] = ProbabilitySimplexParameter,
        concentration: Union[Tensor, float],
        num_classes: Optional[int] = None,
        axis: int = -1,
        where: Union[str, Callable] = "weight",
        key: jax.random.PRNGKey,
        **params,
    ) -> PyTree:
        # TODO: This is a hack to get around the fact that the initialiser uses
        #       the same name for the parameter as the mapper. We need a better
        #       solution; this only fixes the problem for the current use case.
        #       e.g., if we subclass ProbabilitySimplexParameter, then this
        #       doesn't work. Or if we use a NormSphereParameter, then it is
        #       impossible to override the default axis value.
        if mapper is ProbabilitySimplexParameter:
            params.update(axis=axis)
        init = cls(
            mapper=mapper,
            concentration=concentration,
            num_classes=num_classes,
            axis=axis,
        )
        return super()._init_impl(
            init=init,
            model=model,
            where=where,
            key=key,
            **params,
        )
