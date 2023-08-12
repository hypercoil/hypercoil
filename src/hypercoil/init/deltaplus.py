# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Initialise parameters as a set of delta functions, plus Gaussian noise.
"""
from __future__ import annotations
from typing import Callable, Optional, Tuple, Type, Union

import jax
import jax.numpy as jnp

from ..engine import PyTree, Tensor
from .base import MappedInitialiser
from .mapparam import MappedParameter


def deltaplus_init(
    *,
    shape: Tuple[int, ...],
    loc: Optional[Tuple[int, ...]] = None,
    scale: float = 1,
    var: Tensor = 0.2,
    key: jax.random.PRNGKey,
):
    """
    Delta-plus initialisation.

    Initialise a tensor as a delta function added to Gaussian noise. This can
    be used to initialise filters for time series convolutions. The
    initialisation can be configured to produce a filter that approximately
    returns the input signal (or a lagged version of the input signal).

    Parameters
    ----------
    shape : tuple
        Shape of the tensor to initialise.
    loc : tuple or None (default None)
        Location of the delta function expressed as an n-tuple of array
        coordinates along the last n axes of the tensor. Defaults to the
        centre of the tensor.
    scale : float (default 1)
        Magnitude of the delta function. Defaults to 1.
    var : float or Tensor (default 0.2)
        Variance of the Gaussian distribution from which the random noise is
        sampled. By default, noise is sampled i.i.d. for all entries in the
        tensor. To change the i.i.d. behaviour, use a tensor of floats that
        is broadcastable to the specified shape.
    key : jax.random.PRNGKey
        Pseudo-random number generator key for sampling the Gaussian noise.

    Returns
    -------
    None. The input tensor is initialised in-place.
    """
    loc = loc or tuple([x // 2 for x in shape])
    val = jnp.zeros(shape)
    val = val.at[(...,) + loc].add(scale)
    noise = jax.random.normal(key, shape=shape) * var
    return val + noise


class DeltaPlusInitialiser(MappedInitialiser):
    """
    Parameter initialiser following the delta-plus-noise scheme.

    Initialise a tensor as a delta function added to Gaussian noise.

    See :func:`deltaplus_init_` and :class:`MappedInitialiser` for usage
    details.
    """

    loc: Optional[Tuple[int, ...]] = None
    scale: float = 1
    var: Tensor = 0.2

    def __init__(
        self,
        loc: Optional[Tuple[int, ...]] = None,
        scale: float = 1,
        var: Tensor = 0.2,
        mapper: Optional[Type[MappedParameter]] = None,
    ):
        self.loc = loc
        self.scale = scale
        self.var = var
        super().__init__(mapper=mapper)

    def _init(
        self,
        shape: Tuple[int, ...],
        key: jax.random.PRNGKey,
    ) -> Tensor:
        return deltaplus_init(
            shape=shape,
            loc=self.loc,
            scale=self.scale,
            var=self.var,
            key=key,
        )

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Optional[Type[MappedParameter]] = None,
        loc: Optional[Tuple[int, ...]] = None,
        scale: float = 1,
        var: Tensor = 0.2,
        where: Union[str, Callable] = 'weight',
        key: jax.random.PRNGKey,
        **params,
    ) -> PyTree:
        init = cls(mapper=mapper, loc=loc, scale=scale, var=var)
        return super()._init_impl(
            init=init,
            model=model,
            where=where,
            key=key,
            **params,
        )
