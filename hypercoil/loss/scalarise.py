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


def document_scalarisation_map(func: Callable) -> Callable:
    """
    Decorator for scalarisation maps to document them.
    """
    param_spec = """
    Parameters
    ----------
    f : Callable[Sequence[Any], Tensor]
        The tensor-valued function to be transformed."""
    norm_spec = """
    p : Any
        The norm order.
    axis : Union[int, Sequence[int]]
        The axis or axes along which to compute the norm."""
    staged_spec = """
    outer_scalarise : Optional[Callable]
        The scalarisation map to use to map the tensor of norms to a scalar.
        If ``None``, then the mean of the norms is used."""
    unused_key_spec = """
    key : Optional[``jax.random.PRNGKey``]
        An optional random number generator key. Unused; exists for
        conformance with other scalarisation maps."""
    return_spec = """
    Returns
    -------
    Callable[Sequence[Any], float]
        The scalar-valued function."""

    func.__doc__ = func.__doc__.format(
        param_spec=param_spec,
        norm_spec=norm_spec,
        staged_spec=staged_spec,
        unused_key_spec=unused_key_spec,
        return_spec=return_spec,
    )
    return func


@document_scalarisation_map
def sum_scalarise(
    f: Callable[..., Tensor] = identity,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Callable[..., float]:
    """
    Transform a tensor-valued function to a scalar-valued function by summing
    the tensor.
    \
    {param_spec}\
    {unused_key_spec}\
    \
    {return_spec}
    """
    def reduced_f(*pparams, **params):
        X = f(*pparams, **params)
        return jnp.sum(X)
    return reduced_f


@document_scalarisation_map
def mean_scalarise(
    f: Callable[..., Tensor] = identity,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Callable[..., float]:
    """
    Transform a tensor-valued function to a scalar-valued function by taking
    the mean of the tensor.
    \
    {param_spec}\
    {unused_key_spec}\
    \
    {return_spec}
    """
    def reduced_f(*pparams, **params):
        X = f(*pparams, **params)
        return jnp.mean(X)
    return reduced_f


@document_scalarisation_map
def meansq_scalarise(
    f: Callable[..., Tensor] = identity,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Callable[..., float]:
    """
    Transform a tensor-valued function to a scalar-valued function by taking
    the mean of the elementwise squared tensor.
    \
    {param_spec}\
    {unused_key_spec}\
    \
    {return_spec}
    """
    def reduced_f(*pparams, **params):
        X = f(*pparams, **params)
        return jnp.mean(X ** 2)
    return reduced_f


def max_scalarise(
    f: Callable[..., Tensor] = identity,
    *,
    axis: Union[int, Sequence[int]] = -1,
    outer_scalarise: Optional[Callable] = None,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Callable[..., float]:
    """
    Transform a tensor-valued function to a scalar-valued function by taking
    the maximum of the tensor.

    This may be more appropriate than using the infinity norm, particularly
    for negative-valued loss functions.
    \
    {param_spec}\
    {unused_key_spec}\
    \
    {return_spec}
    """
    def reduced_f(*pparams, **params):
        X = f(*pparams, **params)
        return jnp.max(X, axis=axis)

    scalarise = outer_scalarise or mean_scalarise
    return scalarise(reduced_f, key=key)


@document_scalarisation_map
def norm_scalarise(
    f: Callable[..., Tensor] = identity,
    *,
    p: Any = 2,
    axis: Union[int, Sequence[int]] = -1,
    force_vector_norm: bool = False,
    outer_scalarise: Optional[Callable] = None,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Callable[..., float]:
    """
    Compute a specified norm along an axis or set of axes, and then map the
    tensor of norms to a scalar using a scalarisation map. This is equivalent
    to a composition of the norm along an axis or set of axes with an outer
    scalarisation map.
    \
    {param_spec}\
    {norm_spec}\
    {staged_spec}\
    force_vector_norm : bool
        If ``True``, then the tensor is unfolded along the specified axis or
        axes before computing the norm. This forces the reduction to be a
        vector norm, rather than a matrix norm, if the number of reduced axes
        is greater than one.
    {unused_key_spec}\
    \
    {return_spec}
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


@document_scalarisation_map
def vnorm_scalarise(
    f: Callable[..., Tensor] = identity,
    *,
    p: Any = 2,
    axis: Union[int, Sequence[int]] = -1,
    outer_scalarise: Optional[Callable] = None,
    key: Optional['jax.random.PRNGKey'] = None,
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
    {param_spec}\
    {norm_spec}\
    {staged_spec}\
    {unused_key_spec}\
    \
    {return_spec}
    """
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


@document_scalarisation_map
def wmean_scalarise(
    f: Callable[..., Tensor] = identity,
    *,
    axis: Union[int, Sequence[int]] = None,
    outer_scalarise: Optional[Callable] = None,
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Transform a tensor-valued function to a scalar-valued function by taking
    the weighted mean of the tensor along an axis or set of axes, and then
    mapping the resulting means to a scalar using a scalarisation map.

    {param_spec}\
    {staged_spec}\
    {unused_key_spec}\
    \
    {return_spec}
    """
    def reduced_f(*pparams, scalarisation_weight, **params):
        X = f(*pparams, **params)
        weight = jnp.linalg.norm(X, ord=2, axis=axis)
        return wmean(X, scalarisation_weight, axis=axis)

    scalarise = outer_scalarise or mean_scalarise
    return scalarise(reduced_f, key=key)


@document_scalarisation_map
def selfwmean_scalarise(
    f: Callable[..., Tensor] = identity,
    *,
    axis: Union[int, Sequence[int]] = None,
    gradpath: Optional[Literal['weight', 'input']] = 'input',
    softmax_axis: Optional[Union[Sequence[int], int, bool]] = False,
    softmax_invert: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Transform a tensor-valued function to a scalar-valued function by taking
    the self-weighted mean of the tensor along an axis or set of axes.

    {param_spec}\
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
        of the maximum.
    {unused_key_spec}\
    \
    {return_spec}
    """
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
