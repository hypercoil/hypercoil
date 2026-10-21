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
    normalise: Optional[Literal['max', 'sum']] = None,
    var: float = 0.02,
    excl_axis: Optional[Sequence[int]] = None,
    key: jax.random.PRNGKey,
) -> Tensor:
    r"""
    Sample a tensor from a separable double exponential function.

    Along each included axis the profile is :math:`e^{-|x - l| / w}` for array
    coordinate :math:`x`, location :math:`l` and width :math:`w`; the
    profiles are multiplied across axes and broadcast to ``shape``.

    Parameters
    ----------
    shape
        Shape of the output tensor.
    loc
        Location of the peak along each of the last ``len(loc)`` axes, in
        array coordinates. Only those axes receive a profile. None places the
        peak at the centre of every axis.
    width
        Decay length along each axis, aligned with the last axes of
        ``shape``. None gives 1 for every axis. A large width approximates a
        constant; a small one, a delta function at ``loc``.
    normalise
        ``'max'`` scales the output to a maximum of 1, ``'sum'`` to a sum of
        1; None leaves the output unscaled. Applied before the noise.
    var
        Standard deviation of the additive Gaussian noise; 0 for none.
    excl_axis
        Axes that receive no profile; the result is broadcast along them.
    key
        Pseudo-random number generator key for the noise.

    Returns
    -------
    Tensor
        Tensor of shape ``shape``.
    """
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
    if normalise == 'max':
        val /= val.max()
    elif normalise == 'sum':
        val /= val.sum()
    val = jnp.broadcast_to(val, shape)
    if var != 0:
        return val + jax.random.normal(key=key, shape=shape) * var
    return val


class LaplaceInitialiser(MappedInitialiser):
    """
    Double exponential initialiser; see :func:`laplace_init`.
    """

    loc: Optional[Sequence[int]]
    width: Optional[Sequence[float]]
    normalise: Optional[Literal['max', 'sum']]
    var: float
    excl_axis: Optional[Sequence[int]]

    def __init__(
        self,
        loc: Optional[Sequence[int]] = None,
        width: Optional[Sequence[float]] = None,
        normalise: Optional[Literal['max', 'sum']] = None,
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
        normalise: Optional[Literal['max', 'sum']] = None,
        var: float = 0.02,
        excl_axis: Optional[Sequence[int]] = None,
        where: Union[str, Callable] = 'weight',
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
    In-place double exponential initialisation; not currently implemented.

    Use :func:`laplace_init` or :class:`LaplaceInitialiser`.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError


class LaplaceInit:
    """
    Double exponential initialisation; not currently implemented.

    Use :class:`LaplaceInitialiser`.
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
