# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Losses
~~~~~~
Loss functions and functionals.

A loss function is the composition of a score function and a scalarisation
map (which might itself be the composition of different tensor rank reduction
maps.)
"""
import jax
import jax.numpy as jnp
from distrax._src.utils.math import mul_exp
from functools import partial, reduce
from typing import Any, Callable, Literal, Optional, Sequence, Tuple, Union

from ..engine import Tensor, promote_axis, standard_axis_number
from ..functional import corr_kernel, recondition_eigenspaces


def identity(
    X: Tensor,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    return X


# Scalarisations -------------------------------------------------------------


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


# Constraint violation penalties ---------------------------------------------


def zero(
    X: Tensor,
    *,
    broadcast: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    if broadcast:
        return jnp.zeros_like(X)
    return 0.


def constraint_violation(
    X: Tensor,
    *,
    constraints: Sequence[Callable[[Tensor], Tensor]],
    broadcast_against_input: bool = False,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    broadcast = broadcast_against_input
    constraints = (partial(zero, broadcast=broadcast),) + tuple(constraints)
    if key is not None:
        return reduce(jnp.maximum, (c(X, key=key) for c in constraints))
    return reduce(jnp.maximum, (c(X) for c in constraints))


def unilateral_loss(
    X: Tensor,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    return constraint_violation(X, constraints=(identity,))


def hinge_loss(
    Y_hat: Tensor,
    Y: Tensor,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    score = 1 - Y * Y_hat
    return constraint_violation(score, constraints=(identity,))


# Smoothness -----------------------------------------------------------------


def smoothness(
    X: Tensor,
    *,
    n: int = 1,
    #pad_value: Optional[Union[float, Literal['initial']]] = None,
    pad_value: Optional[float] = None,
    axis: int = -1,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tensor:
    # if pad_value == 'initial':
    #     axis = standard_axis_number(axis, X.ndim)
    #     pad_value = X[(slice(None),) * axis + (0,)]
    return jnp.diff(X, n=n, axis=axis, prepend=pad_value)


# Bimodal symmetric ----------------------------------------------------------


def bimodal_symmetric(
    X: Tensor,
    *,
    modes: Tuple[int, int] = (0, 1),
    key: Optional['jax.random.PRNGKey'] = None,
):
    mean = sum(modes) / 2
    step = max(modes) - mean
    return jnp.abs(jnp.abs((X - mean)) - step)


# Gramian determinants -------------------------------------------------------


def det_gram(
    X: Tensor,
    theta: Optional[Tensor] = None,
    *,
    op: Optional[Callable] = corr_kernel,
    psi: Optional[float] = 0.,
    xi: Optional[float] = 0.,
    key: Optional['jax.random.PRNGKey'] = None,
):
    Z = op(X, theta=theta)
    if xi > 0:
        Z = recondition_eigenspaces(Z, psi=psi, xi=xi, key=key)
    elif psi > 0:
        Z = Z + psi * jnp.eye(Z.shape[-1])
    return -jnp.linalg.det(Z)


def log_det_gram(
    X: Tensor,
    theta: Optional[Tensor] = None,
    *,
    op: Optional[Callable] = corr_kernel,
    psi: Optional[float] = 0.,
    xi: Optional[float] = 0.,
    key: Optional['jax.random.PRNGKey'] = None,
):
    Z = op(X, theta=theta)
    if xi > 0:
        Z = recondition_eigenspaces(Z, psi=psi, xi=xi, key=key)
    elif psi > 0:
        Z = Z + psi * jnp.eye(Z.shape[-1])
    _, logdet = jnp.linalg.slogdet(Z)
    return -logdet


# Information and entropy ----------------------------------------------------


def entropy(
    X: Tensor,
    *,
    axis: Union[int, Sequence[int]] = -1,
    keepdims: bool = True,
    reduce: bool = True,
) -> Tensor:
    """
    Compute the entropy of a categorical distribution.
    """
    eps = jnp.finfo(X.dtype).eps
    entropy = -X * jnp.log(X + eps)
    if not reduce:
        return entropy
    return entropy.sum(axis, keepdims=keepdims)


def entropy_logit(
    X: Tensor,
    *,
    temperature: float = 1.,
    axis: Union[int, Sequence[int]] = -1,
    keepdims: bool = True,
    reduce: bool = True,
) -> Tensor:
    """
    Project logits in the input matrix onto the probability simplex, and then
    compute the entropy of the resulting categorical distribution.
    """
    probs = jax.nn.softmax(X / temperature, axis=axis)
    return entropy(probs, axis=axis, keepdims=keepdims, reduce=reduce)


def kl_divergence(
    P: Tensor,
    Q: Tensor,
    *,
    axis: Union[int, Sequence[int]] = -1,
    keepdims: bool = True,
    reduce: bool = True,
) -> Tensor:
    """Adapted from distrax."""
    eps = jnp.finfo(P.dtype).eps
    P = jnp.log(P + eps)
    Q = jnp.log(Q + eps)
    kl_div = mul_exp(P - Q, P)
    if not reduce:
        return kl_div
    return kl_div.sum(axis=axis, keepdims=keepdims)


def kl_divergence_logit(
    P: Tensor,
    Q: Tensor,
    *,
    axis: Union[int, Sequence[int]] = -1,
    keepdims: bool = True,
    reduce: bool = True,
):
    """Adapted from distrax."""
    P = jax.nn.log_softmax(P, axis=axis)
    Q = jax.nn.log_softmax(Q, axis=axis)
    kl_div = mul_exp(P - Q, P)
    if not reduce:
        return kl_div
    return kl_div.sum(axis=axis, keepdims=keepdims)


def js_divergence(
    P: Tensor,
    Q: Tensor,
    *,
    axis: Union[int, Sequence[int]] = -1,
    keepdims: bool = True,
    reduce: bool = True,
) -> Tensor:
    M = 0.5 * (P + Q)
    js_div = (kl_divergence(P, M, reduce=False) +
              kl_divergence(Q, M, reduce=False)) / 2
    if not reduce:
        return js_div
    return js_div.sum(axis=axis, keepdims=keepdims)


def js_divergence_logit(
    P: Tensor,
    Q: Tensor,
    *,
    axis: Union[int, Sequence[int]] = -1,
    keepdims: bool = True,
    reduce: bool = True,
) -> Tensor:
    prob_axis = axis
    if prob_axis is None:
        prob_axis = -1
    P = jax.nn.softmax(P, prob_axis)
    Q = jax.nn.softmax(Q, prob_axis)
    return js_divergence(P, Q, axis=axis, keepdims=keepdims, reduce=reduce)


# Equilibrium ----------------------------------------------------------------


def equilibrium(
    X: Tensor,
    *,
    level_axis: Union[int, Sequence[int]] = -1,
    instance_axes: Union[int, Sequence[int]] = (-1, -2),
    keepdims: bool = True,
) -> Tensor:
    """
    Compute the parcel equilibrium.
    """
    parcel = X.mean(level_axis, keepdims=keepdims)
    total = X.mean(instance_axes, keepdims=keepdims)
    return parcel - total


def equilibrium_logit(
    X: Tensor,
    *,
    level_axis: Union[int, Sequence[int]] = -1,
    prob_axis: Union[int, Sequence[int]] = -2,
    instance_axes: Union[int, Sequence[int]] = (-1, -2),
    keepdims: bool = True,
) -> Tensor:
    """
    Project logits in the input matrix onto the probability simplex, and then
    compute the parcel equilibrium.
    """
    probs = jax.nn.softmax(X, axis=prob_axis)
    return equilibrium(
        probs,
        level_axis=level_axis,
        instance_axes=instance_axes,
        keepdims=keepdims,
    )
