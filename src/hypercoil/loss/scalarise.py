# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Scalarisation (or rank-reduction) maps for loss functions.

A loss function is the composition of a score function and a scalarisation
(which might itself be the composition of different tensor rank reduction
maps.)
"""
from __future__ import annotations
from typing import Any, Callable, Literal, Optional, Sequence, Union

import jax
import jax.numpy as jnp

from ..engine import (
    NestedDocParse,
    Tensor,
    promote_axis,
    standard_axis_number,
)
from .functional import identity


def document_scalarisation_map(func: Callable) -> Callable:
    """
    Decorator for documenting scalarisation maps.
    """
    long_description_scalarise = """
    .. note::

        This function is a scalarisation map, which can be either used on its
        own or composed with other scalarisation maps to form a scalarisation.

        Composition is performed by passing the scalarisation map to be
        composed as the ``inner`` argument to the outer scalarisation map.
        This can be chained indefinitely.

        The **output** of a scalarisation map (or composition thereof) is a
        function that maps a tensor-valued function to a scalar-valued
        function. Be mindful of the signature of scalarisation maps: they do
        not themselves perform this map, but rather return a function that
        does.

        For example, the following are all valid scalarisations of a function
        ``f``::

            max_scalarise()(f)
            mean_scalarise(inner=max_scalarise(axis=-1))(f)
            mean_scalarise(inner=vnorm_scalarise(p=1, axis=(0, 2)))(f)

        Calling the scalarised function will return a scalar value.
        """
    param_spec = """
    Parameters
    ----------
    inner : Callable, optional
        The inner scalarisation map. If not specified, the identity map is
        used. For many scalarisation maps, the default settings for the inner
        map and the scalarisation axis amount to applying the scalarisation
        map over the entire tensor. Users are advised to verify the default
        settings for the inner map and the scalarisation axis.
    axis: Union[int, Sequence[int]], optional
        The axis or axes over which to apply the scalarisation map. If not
        specified, the scalarisation map is applied over the entire tensor,
        except in the cases of maps that are only defined over vectors or
        matrices (e.g., norm scalarisations). Check the default arguments to
        verify the default behaviour.

        .. warning::

            When composing scalarisation maps, the value of the ``axis``
            argument will refer to the axis or axes of the reduced-rank tensor
            that is the output of the inner scalarisation map. For example,
            consider the following composition::

                mean_scalarise(inner=max_scalarise(axis=-1), axis=-2)(f)

            The ``axis`` argument of the outer scalarisation map refers to the
            **third** from last axis of the original tensor, which is the
            second from last axis of the tensor after the inner scalarisation
            map has been applied.

            Setting ``keepdims=True`` for the inner scalarisations will result
            in the (perhaps more intuitive) behaviour of the ``axis`` argument
            referring to the axis or axes of the original tensor.

    keepdims: bool, optional
        Whether to keep the reduced dimensions in the output. If ``True``, the
        output will have the same number of dimensions as the input, each of
        singleton size. If ``False``, the output will have one fewer dimension
        than the input for each dimension over which the scalarisation map is
        applied.

        .. note::

            It can be useful to set ``keepdims=True`` when composing
            scalarisation maps, as the ``axis`` argument can then be specified
            with reference to the original tensor rather than the reduced
            tensor, only setting ``keepdims=False`` for the outermost
            scalarisation map."""
    norm_spec = """
    p : Any
        The norm order."""
    unused_key_spec = """
    key : Optional[``jax.random.PRNGKey``]
        An optional random number generator key. Unused; exists for
        conformance with potential future scalarisation maps that could inject
        randomness."""
    return_spec = """
    Returns
    -------
    Callable[[Callable[..., Tensor]], Callable[..., float]]
        The scalarisation transformation. This is a function that takes a
        tensor-valued function and returns a scalar-valued function."""

    fmt = NestedDocParse(
        long_description_scalarise=long_description_scalarise,
        param_spec=param_spec,
        norm_spec=norm_spec,
        unused_key_spec=unused_key_spec,
        return_spec=return_spec,
    )
    func.__doc__ = func.__doc__.format_map(fmt)
    return func


@document_scalarisation_map
def sum_scalarise(
    *,
    inner: Optional[Callable] = None,
    axis: Union[int, Sequence[int]] = None,
    keepdims: bool = False,
    key: Optional["jax.random.PRNGKey"] = None,
) -> Callable[[Callable[..., Tensor]], Callable[..., float]]:
    """
    Transform a tensor-valued function to a scalar-valued function by summing
    the tensor.
    \
    {long_description_scalarise}
    \
    {param_spec}\
    {unused_key_spec}
    \
    {return_spec}
    """
    if inner is None:
        inner = identity

    def scalarisation(f: Callable[..., Tensor] = identity):
        def reduced_f(*pparams, **params):
            X = inner(f)(*pparams, **params)
            return jnp.sum(X, axis=axis, keepdims=keepdims)

        return reduced_f

    return scalarisation


@document_scalarisation_map
def mean_scalarise(
    *,
    inner: Optional[Callable] = None,
    axis: Union[int, Sequence[int]] = None,
    keepdims: bool = False,
    key: Optional["jax.random.PRNGKey"] = None,
) -> Callable[[Callable[..., Tensor]], Callable[..., float]]:
    """
    Transform a tensor-valued function to a scalar-valued function by taking
    the mean of the tensor.
    \
    {long_description_scalarise}
    \
    {param_spec}\
    {unused_key_spec}
    \
    {return_spec}
    """
    if inner is None:
        inner = identity

    def scalarisation(f: Callable[..., Tensor] = identity):
        def reduced_f(*pparams, **params):
            X = inner(f)(*pparams, **params)
            return jnp.mean(X, axis=axis, keepdims=keepdims)

        return reduced_f

    return scalarisation


@document_scalarisation_map
def meansq_scalarise(
    *,
    inner: Optional[Callable] = None,
    axis: Union[int, Sequence[int]] = None,
    keepdims: bool = False,
    key: Optional["jax.random.PRNGKey"] = None,
) -> Callable[[Callable[..., Tensor]], Callable[..., float]]:
    """
    Transform a tensor-valued function to a scalar-valued function by taking
    the mean of the elementwise squared tensor.
    \
    {long_description_scalarise}
    \
    {param_spec}\
    {unused_key_spec}
    \
    {return_spec}
    """
    if inner is None:
        inner = identity

    def scalarisation(f: Callable[..., Tensor] = identity):
        def reduced_f(*pparams, **params):
            X = inner(f)(*pparams, **params)
            return jnp.mean(X**2, axis=axis, keepdims=keepdims)

        return reduced_f

    return scalarisation


@document_scalarisation_map
def max_scalarise(
    *,
    inner: Optional[Callable] = None,
    axis: Union[int, Sequence[int]] = None,
    keepdims: bool = False,
    key: Optional["jax.random.PRNGKey"] = None,
) -> Callable[[Callable[..., Tensor]], Callable[..., float]]:
    """
    Transform a tensor-valued function to a scalar-valued function by taking
    the maximum of the tensor.

    This may be more appropriate than using the infinity norm, particularly
    for negative-valued loss functions.
    \
    {long_description_scalarise}
    \
    {param_spec}\
    {unused_key_spec}
    \
    {return_spec}
    """
    if inner is None:
        inner = identity

    def scalarisation(f: Callable[..., Tensor] = identity):
        def reduced_f(*pparams, **params):
            X = inner(f)(*pparams, **params)
            return jnp.max(X, axis=axis, keepdims=keepdims)

        return reduced_f

    return scalarisation


@document_scalarisation_map
def norm_scalarise(
    *,
    p: Any = 2,
    force_vector_norm: bool = False,
    axis: Union[int, Sequence[int]] = -1,
    inner: Optional[Callable] = None,
    keepdims: bool = False,
    key: Optional["jax.random.PRNGKey"] = None,
) -> Callable[..., float]:
    """
    Compute a specified norm along an axis or set of axes, and then map the
    tensor of norms to a scalar using a scalarisation map. This is equivalent
    to a composition of the norm along an axis or set of axes with an outer
    scalarisation map.
    \
    {long_description_scalarise}
    \
    {param_spec}\
    {norm_spec}
    force_vector_norm : bool
        If ``True``, then the tensor is unfolded along the specified axis or
        axes before computing the norm. This forces the reduction to be a
        vector norm, rather than a matrix norm, if the number of reduced axes
        is greater than one.\
    {unused_key_spec}
    \
    {return_spec}
    """
    if inner is None:
        inner = identity

    def scalarisation(f: Callable[..., Tensor] = identity):
        def reduced_f(*pparams, **params):
            X = inner(f)(*pparams, **params)
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
                    norm = jnp.linalg.norm(
                        Xperm, ord=p, axis=0, keepdims=keepdims
                    )
                    return norm
                norm = jnp.linalg.norm(X, ord=p, axis=axes, keepdims=keepdims)
                return norm
            norm = jnp.linalg.norm(X, ord=p, axis=axis, keepdims=keepdims)
            return norm

        return reduced_f

    return scalarisation


@document_scalarisation_map
def vnorm_scalarise(
    *,
    p: Any = 2,
    axis: Union[int, Sequence[int]] = -1,
    inner: Optional[Callable] = None,
    keepdims: bool = False,
    key: Optional["jax.random.PRNGKey"] = None,
) -> Callable[..., float]:
    """
    Transform a tensor-valued function to a scalar-valued function by taking
    the vector norm of the tensor along an axis or set of axes, and then
    mapping the resulting norms to a scalar using a scalarisation map.

    Like :func:`norm_scalarise`, but always unfolds the tensor along the
    specified axis or axes before computing the norm so that the norm is
    always a vector norm rather than a matrix norm. This is equivalent to
    using :func:`norm_scalarise` with ``force_vector_norm=True``.
    \
    {long_description_scalarise}
    \
    {param_spec}\
    {norm_spec}\
    {unused_key_spec}
    \
    {return_spec}
    """
    return norm_scalarise(
        p=p,
        force_vector_norm=True,
        axis=axis,
        inner=inner,
        keepdims=keepdims,
        key=key,
    )


def wmean(
    input: Tensor,
    weight: Tensor,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: bool = False,
) -> Tensor:
    """
    Rank-reducing function for scalarisation maps: weighted mean.

    >>> wmean(jnp.array([1, 2, 3]), jnp.array([1, 0, 1]))
    Array(2., dtype=float32)

    >>> wmean(
    ...     jnp.array([[1, 2, 3],
    ...                [1, 2, 3],
    ...                [1, 2, 3]]),
    ...     jnp.array([1, 0, 1]),
    ...     axis=0
    ... )
    Array([1., 2., 3.], dtype=float32)

    >>> wmean(
    ...     jnp.array([[1, 2, 3],
    ...                [1, 2, 3],
    ...                [1, 2, 3]]),
    ...     jnp.array([1, 0, 1]),
    ...     axis=1,
    ...     keepdims=True
    ... )
    Array([[2.],
                 [2.],
                 [2.]], dtype=float32)
    """
    if axis is None:
        axis = tuple(range(input.ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    assert weight.ndim == len(
        axis
    ), "Weight must have as many dimensions as are being reduced"
    retain = [(i not in axis) for i in range(input.ndim)]
    for i, d in enumerate(retain):
        if d:
            weight = jnp.expand_dims(weight, i)
    wtd = weight * input
    num = wtd.sum(axis, keepdims=keepdims)
    denom = weight.sum(axis, keepdims=keepdims)
    return num / denom


def selfwmean(
    input: Tensor,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: bool = False,
    gradpath: Optional[Literal["weight", "input"]] = "input",
    softmax_axis: Optional[Union[Sequence[int], int, bool]] = False,
    softmax_invert: bool = False,
) -> Tensor:
    """
    Self-weighted mean rank-reducing function. With the softmax turned on,
    this should be close to a soft version of the maximum.
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
    if gradpath == "input":
        weight = jax.lax.stop_gradient(weight)
    elif gradpath == "weight":
        input = jax.lax.stop_gradient(input)
    return wmean(
        input=input,
        weight=weight,
        axis=axis,
        keepdims=keepdims,
    )


@document_scalarisation_map
def wmean_scalarise(
    *,
    inner: Optional[Callable] = None,
    axis: Union[int, Sequence[int]] = None,
    keepdims: bool = False,
    key: Optional["jax.random.PRNGKey"] = None,
) -> Callable[[Callable[..., Tensor]], Callable[..., float]]:
    """
    Transform a tensor-valued function to a scalar-valued function by taking
    the weighted mean of the tensor along an axis or set of axes, and then
    mapping the resulting means to a scalar using a scalarisation map.

    .. warning::

        Nesting two or more ``wmean_scalarise`` scalarisation maps together
        is not supported. Instead, broadcast multiply the scalarisation
        weights across the axes to be reduced and use a single
        ``wmean_scalarise`` scalarisation map.

        The exception to this is when the scalarisation weights are the same
        for all axes to be reduced by ``wmean_scalarise`` maps.
    \
    {long_description_scalarise}
    \
    {param_spec}\
    {unused_key_spec}
    \
    {return_spec}
    """
    if inner is None:
        inner = identity

    def scalarisation(f: Callable[..., Tensor] = identity):
        def reduced_f(*pparams, scalarisation_weight, **params):
            X = inner(f)(*pparams, **params)
            return wmean(
                X,
                scalarisation_weight,
                axis=axis,
                keepdims=keepdims,
            )

        return reduced_f

    return scalarisation


@document_scalarisation_map
def selfwmean_scalarise(
    *,
    inner: Optional[Callable] = None,
    axis: Union[int, Sequence[int]] = None,
    gradpath: Optional[Literal["weight", "input"]] = "input",
    softmax_axis: Optional[Union[Sequence[int], int, bool]] = False,
    softmax_invert: bool = False,
    keepdims: bool = False,
    key: Optional["jax.random.PRNGKey"] = None,
) -> Callable[[Callable[..., Tensor]], Callable[..., float]]:
    """
    Transform a tensor-valued function to a scalar-valued function by taking
    the self-weighted mean of the tensor along an axis or set of axes.
    \
    {long_description_scalarise}
    \
    {param_spec}
    gradpath: Optional[Literal['weight', 'input']] (default: 'input')
        If 'weight', the gradient of the scalarisation function will be
        backpropagated through the weights only. If 'input', the gradient of
        the scalarisation function will be backpropagated through the input
        only. If None, the gradient will be backpropagated through both.
    softmax_axis: Optional[Union[Sequence[int], int, bool]] (default: False)
        If not False, instead of using the input as the weight, the input is
        passed through a softmax function to create a weight. If True, the
        softmax is taken over all axes. If an integer or sequence of integers,
        the softmax is taken over those axes. If False, the input is used as
        the weight.
    softmax_invert: bool (default: False)
        If True, the input is negated before passing it through the softmax.
        In this way, the softmax can be used to upweight the minimum instead
        of the maximum.\
    {unused_key_spec}
    \
    {return_spec}
    """
    if inner is None:
        inner = identity

    def scalarisation(f: Callable[..., Tensor] = identity):
        def reduced_f(*pparams, **params):
            X = inner(f)(*pparams, **params)
            return selfwmean(
                X,
                axis=axis,
                gradpath=gradpath,
                softmax_axis=softmax_axis,
                softmax_invert=softmax_invert,
                keepdims=keepdims,
            )

        return reduced_f

    return scalarisation
