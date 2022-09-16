# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Scalarisation maps for loss functions.

A loss function is the composition of a score function and a scalarisation
map (which might itself be the composition of different tensor rank reduction
maps.)
"""
import jax
import jax.numpy as jnp
from typing import Any, Callable, Literal, Optional, Sequence, Union
from .functional import identity
from ..engine import Tensor, promote_axis, standard_axis_number


def sum_scalarise(
    f: Callable[Sequence[Any], Tensor] = identity,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Callable[Sequence[Any], float]:
    def reduced_f(*pparams, **params):
        X = f(*pparams, **params)
        return jnp.sum(X)
    return reduced_f


def mean_scalarise(
    f: Callable[Sequence[Any], Tensor] = identity,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Callable[Sequence[Any], float]:
    def reduced_f(*pparams, **params):
        X = f(*pparams, **params)
        return jnp.mean(X)
    return reduced_f


def meansq_scalarise(
    f: Callable[Sequence[Any], Tensor] = identity,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Callable[Sequence[Any], float]:
    def reduced_f(*pparams, **params):
        X = f(*pparams, **params)
        return jnp.mean(X ** 2)
    return reduced_f


def norm_scalarise(
    f: Callable[Sequence[Any], Tensor] = identity,
    *,
    p: Any = 2,
    axis: Union[int, Sequence[int]] = -1,
    force_vector_norm: bool = False,
    outer_scalarise: Optional[Callable] = None,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Callable[Sequence[Any], float]:
    """
    Compute a specified norm along an axis or set of axes, and then map the
    tensor of norms to a scalar using a scalarisation map. This is equivalent
    to a composition of the norm along an axis or set of axes with an outer
    scalarisation map.
    """
    def reduced_f(*pparams, **params):
        X = f(*pparams, **params)
        if force_vector_norm and not isinstance(axis, int):
            if axis is None:
                axes = tuple(range(X.ndim))
            else:
                axes = axis
            ndim_norm = len(axes)
            ndim = X.ndim
            if ndim_norm > 1:
                axes = tuple(standard_axis_number(ax, ndim) for ax in axes)
                Xperm = X.transpose(promote_axis(ndim, axes))
                Xperm = Xperm.reshape(-1, *Xperm.shape[ndim_norm:])
                norm = jnp.linalg.norm(Xperm, ord=p, axis=0)
                return norm
            norm = jnp.linalg.norm(X, ord=p, axis=axes)
            return norm
        norm = jnp.linalg.norm(X, ord=p, axis=axis)
        return norm

    scalarise = outer_scalarise or mean_scalarise
    return scalarise(reduced_f, key=key)


def vnorm_scalarise(
    f: Callable[Sequence[Any], Tensor] = identity,
    *,
    p: Any = 2,
    axis: Union[int, Sequence[int]] = -1,
    outer_scalarise: Optional[Callable] = None,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Callable[Sequence[Any], float]:
    return norm_scalarise(
        f,
        p=p,
        axis=axis,
        force_vector_norm=True,
        outer_scalarise=outer_scalarise,
        key=key,
    )


def wmean(
    input: Tensor,
    weight: Tensor,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: bool = False,
) -> Tensor:
    """
    Reducing function for reducing losses: weighted mean.

    >>> wmean(jnp.array([1, 2, 3]), jnp.array([1, 0, 1]))
    DeviceArray(2., dtype=float32)

    >>> wmean(
    ...     jnp.array([[1, 2, 3],
    ...                [1, 2, 3],
    ...                [1, 2, 3]]),
    ...     jnp.array([1, 0, 1]),
    ...     axis=0
    ... )
    DeviceArray([1., 2., 3.], dtype=float32)

    >>> wmean(
    ...     jnp.array([[1, 2, 3],
    ...                [1, 2, 3],
    ...                [1, 2, 3]]),
    ...     jnp.array([1, 0, 1]),
    ...     axis=1,
    ...     keepdims=True
    ... )
    DeviceArray([[2.],
                 [2.],
                 [2.]], dtype=float32)
    """
    if axis is None:
        axis = tuple(range(input.ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    assert weight.ndim == len(axis), (
        'Weight must have as many dimensions as are being reduced')
    retain = [(i not in axis) for i in range(input.ndim)]
    for i, d in enumerate(retain):
        if d: weight = jnp.expand_dims(weight, i)
    wtd = (weight * input)
    return wtd.sum(axis, keepdims=keepdims) / weight.sum(axis, keepdims=keepdims)


def selfwmean(
    input: Tensor,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: bool = False,
    gradpath: Optional[Literal['weight', 'input']] = 'input',
    softmax_axis: Optional[Union[Sequence[int], int, bool]] = False,
    softmax_invert: bool = False,
) -> Tensor:
    """
    Self-weighted mean reducing function. With the softmax turned on, this
    should be close to a soft version of the maximum.
    """
    if softmax_axis is False:
        weight = input
    else:
        if softmax_axis is True:
            softmax_axis = None
        if softmax_invert:
            input = -input
        weight = jax.nn.softmax(input, axis=softmax_axis)
    # I don't think this actually does what we want it to, but this function
    # is actually unsupported, so we won't worry about it yet.
    if gradpath == 'input':
        weight = jax.lax.stop_gradient(weight)
    elif gradpath == 'weight':
        input = jax.lax.stop_gradient(input)
    return wmean(
        input=input,
        weight=weight,
        axis=axis,
        keepdims=keepdims,
    )


def wmean_scalarise(
    f: Callable[Sequence[Any], Tensor] = identity,
    *,
    axis: Union[int, Sequence[int]] = None,
    outer_scalarise: Optional[Callable] = None,
    key: Optional['jax.random.PRNGKey'] = None,
):
    def reduced_f(*pparams, scalarisation_weight, **params):
        X = f(*pparams, **params)
        weight = jnp.linalg.norm(X, ord=2, axis=axis)
        return wmean(X, scalarisation_weight, axis=axis)

    scalarise = outer_scalarise or mean_scalarise
    return scalarise(reduced_f, key=key)


def selfwmean_scalarise(
    f: Callable[Sequence[Any], Tensor] = identity,
    *,
    axis: Union[int, Sequence[int]] = None,
    gradpath: Optional[Literal['weight', 'input']] = 'input',
    softmax_axis: Optional[Union[Sequence[int], int, bool]] = False,
    softmax_invert: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
):
    def reduced_f(*pparams, **params):
        X = f(*pparams, **params)
        return selfwmean(
            X,
            axis=axis,
            gradpath=gradpath,
            softmax_axis=softmax_axis,
            softmax_invert=softmax_invert,
        )

    return reduced_f
