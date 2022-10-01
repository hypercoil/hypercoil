# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Initialise parameters to match a double exponential function.
"""
from __future__ import annotations
from functools import reduce
from typing import (
    Callable,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import jax
import jax.numpy as jnp

from ..engine import PyTree, Tensor
from .base import MappedInitialiser
from .mapparam import MappedParameter


def laplace_init(
    *,
    shape: Tuple[int, ...],
    loc: Optional[Sequence[int]] = None,
    width: Optional[Sequence[float]] = 1,
    normalise: Optional[Literal["max", "sum"]] = None,
    var: float = 0.02,
    excl_axis: Optional[Sequence[int]] = None,
    key: jax.random.PRNGKey,
) -> Tensor:
    if loc is None:
        loc = [(x - 1) / 2 for x in shape]
    if width is None:
        width = [1.0 for _ in range(len(shape))]
    ndim = len(loc)
    if excl_axis is None:
        excl_axis = ()

    axes = []
    for ax, l, w in zip(shape[-ndim:], loc, width[-ndim:]):
        new_ax = jnp.arange(
            -l,
            -l + ax,
        )
        new_ax = jnp.exp(-jnp.abs(new_ax) / w)
        axes += [new_ax]

    ax_shape = [-1]
    val = []
    for i, ax in enumerate(reversed(axes)):
        if -(i + 1) not in excl_axis and (ndim - i - 1) not in excl_axis:
            val = [ax.reshape(ax_shape)] + val
        ax_shape += [1]
    val = reduce(jnp.multiply, val)
    if normalise == "max":
        val /= val.max()
    elif normalise == "sum":
        val /= val.sum()
    val = jnp.broadcast_to(val, shape)
    if var != 0:
        return val + jax.random.normal(key=key, shape=shape) * var
    return val


class LaplaceInitialiser(MappedInitialiser):

    loc: Optional[Sequence[int]]
    width: Optional[Sequence[float]]
    normalise: Optional[Literal["max", "sum"]]
    var: float
    excl_axis: Optional[Sequence[int]]

    def __init__(
        self,
        loc: Optional[Sequence[int]] = None,
        width: Optional[Sequence[float]] = None,
        normalise: Optional[Literal["max", "sum"]] = None,
        var: float = 0.02,
        excl_axis: Optional[Sequence[int]] = None,
        mapper: Optional[Type[MappedParameter]] = None,
    ):
        self.loc = loc
        self.width = width
        self.normalise = normalise
        self.var = var
        self.excl_axis = excl_axis
        super().__init__(mapper=mapper)

    def _init(
        self,
        shape: Tuple[int, ...],
        key: jax.random.PRNGKey,
    ) -> Tensor:
        return laplace_init(
            shape=shape,
            loc=self.loc,
            width=self.width,
            normalise=self.normalise,
            var=self.var,
            excl_axis=self.excl_axis,
            key=key,
        )

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Optional[Type[MappedParameter]] = None,
        loc: Optional[Sequence[int]] = None,
        width: Optional[Sequence[float]] = None,
        normalise: Optional[Literal["max", "sum"]] = None,
        var: float = 0.02,
        excl_axis: Optional[Sequence[int]] = None,
        where: Union[str, Callable] = "weight",
        key: jax.random.PRNGKey = None,
        **params,
    ) -> PyTree:
        init = cls(
            loc=loc,
            width=width,
            normalise=normalise,
            var=var,
            excl_axis=excl_axis,
            mapper=mapper,
        )
        return super()._init_impl(
            init=init,
            model=model,
            where=where,
            key=key,
            **params,
        )


def laplace_init_(
    tensor,
    loc=None,
    width=None,
    norm=None,
    var=0.02,
    excl_axis=None,
    domain=None,
):
    """
    Laplace initialisation.

    Initialise a tensor such that its values are interpolated by a
    multidimensional double exponential function, :math:`e^{-|x|}`.

    Parameters
    ----------
    tensor : Tensor
        Tensor to initialise in-place.
    loc : iterable or None (default None)
        Origin point of the double exponential, in array coordinates. If None,
        this will be set to the centre of the array.
    width : iterable or None (default None)
        Decay rate of the double exponential along each array axis. If None,
        this will be set to 1 isotropically. If this is very large, the result
        will approximate a delta function at the specified ``loc``.
    norm : ``'max'``, ``'sum'``, or None (default None)
        Normalisation to apply to the output.

        - ``'max'`` divides the output by its maximum value such that the
          largest value in the initialised tensor is exactly 1.
        - ``'sum'`` divides the output by its sum such that all entries in the
          initialised tensor sum to 1.
        - None indicates that the output should not be normalised.
    var : float
        Variance of the Gaussian distribution from which the random noise is
        sampled.
    excl_axis : list or None (default None)
        List of axes across which a double exponential is not computed. Instead
        the double exponential computed across the remaining axes is broadcast
        across the excluded axes.
    domain : Domain object (default :doc:`Identity <hypercoil.init.domainbase.Identity>`)
        Used in conjunction with an activation function to constrain or
        transform the values of the initialised tensor. For instance, using
        the :doc:`Atanh <hypercoil.init.domain.Atanh>`
        domain with default scale constrains the tensor as seen by
        data to the range of the tanh function, (-1, 1). Domain objects can
        be used with compatible modules and are documented further in
        :doc:`hypercoil.init.domain <hypercoil.init.domain>`.
        If no domain is specified, the Identity
        domain is used, which does not apply any transformations or
        constraints.

    Returns
    -------
    None. The input tensor is initialised in-place.
    """
    raise NotImplementedError


class LaplaceInit:
    """
    Double exponential initialisation.

    Initialise a tensor such that its values are interpolated by a
    multidimensional double exponential function, :math:`e^{-|x|}`.

    See :func:`laplace_init_` for argument details.
    """

    def __init__(
        self,
        loc=None,
        width=None,
        norm=None,
        var=0.02,
        excl_axis=None,
        domain=None,
    ):
        raise NotImplementedError
