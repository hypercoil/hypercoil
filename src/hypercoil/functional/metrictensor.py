# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Parameterise and integrate over a p-dimensional metric tensor field.

We use these operations to parameterise a latent connectopic space as a
Riemannian manifold. The metric tensor field is a p-dimensional tensor field
that defines the Riemannian metric at each point in the manifold.
"""
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from ..engine import Tensor
from .matrix import diag_embed


def metric_tensor_field_diag_plus_low_rank(
    coor: Tensor,
    low_rank_model: Callable[[Tensor], Tensor],
    diag_model: Callable[[Tensor], Tensor],
) -> Tensor:
    """
    Parameterise a metric tensor field as a sum of a diagonal and a low-rank
    component.

    Parameters
    ----------
    coor : Tensor
        Tensor containing the coordinates of each point in the metric tensor
        field. This is a tensor of shape `(*, p)`, where `*` denotes any number
        of batch and intervening dimensions, and `p` is the number of
        dimensions of the metric tensor field.
    low_rank_model : callable
        Callable that takes a tensor of shape `(*, p)` and returns a tensor of
        shape `(*, p, r)` where `r` is the rank of the low-rank component of
        the metric tensor field.
    diag_model : callable
        Callable that takes a tensor of shape `(*, p)` and returns a tensor of
        shape `(*, p)` containing the diagonal of the metric tensor field.

    Returns
    -------
    Tensor
        Tensor containing the metric tensor at each `coor`. This is a tensor
        of shape `(*, p, p)`.
    """
    low_rank = low_rank_model(coor)
    diag = diag_model(coor)
    return low_rank @ low_rank.swapaxes(-1, -2) + diag_embed(diag)


def quadratic_form(
    coor: Tensor,
    metric_tensor: Tensor,
):
    """
    Compute the quadratic form :math:`x^T A x`.

    Parameters
    ----------
    coor : Tensor
        Tensor of shape `(*, p)` containing the vector to multiply.
    metric_tensor : Tensor
        Tensor of shape `(*, p, p)` containing the metric tensor.

    Returns
    -------
    Tensor
        Tensor of shape `(*, 1)` containing the result of the quadratic form.
    """
    norm = coor[..., None, :] @ metric_tensor @ coor[..., None]
    return norm.squeeze(-1)


def quadratic_form_low_rank_plus_diag(
    coor: Tensor,
    metric_tensor: Tuple[Tensor, Tensor],
) -> Tensor:
    """
    Compute the quadratic form :math:`x^T A x` where :math:`A` is a sum of a
    diagonal and a low-rank component.

    Parameters
    ----------
    coor : Tensor
        Tensor of shape `(*, p)` containing the vector to multiply.
    metric_tensor : tuple of Tensors
        Tuple containing the low-rank and diagonal components of the metric
        tensor. The low-rank component is a tensor of shape `(*, p, r)` and the
        diagonal component is a tensor of shape `(*, p)`. The sum of these two
        tensors is the metric tensor.

    Returns
    -------
    Tensor
        Tensor of shape `(*, 1)` containing the result of the quadratic form.
    """
    low_rank, diag = metric_tensor
    lr = ((low_rank.swapaxes(-1, -2) @ coor[..., None]) ** 2).sum(-2)
    return lr + (diag * (coor ** 2)).sum(-1, keepdims=True)


def _obtain_relative_samples(
    n_samples: int,
    *,
    even_sampling: bool = True,
    include_endpoints: bool = True,
    key: 'jax.random.PRNGKey',
) -> Tensor:
    if even_sampling:
        if include_endpoints:
            rel = jnp.arange(n_samples) / (n_samples - 1)
        else:
            rel = jnp.arange(1, n_samples + 1) / (n_samples + 2)
    else:
        rel = jax.random.uniform(shape=(n_samples,), key=key)
    return rel


def _obtain_absolute_samples(
    a: Tensor,
    b: Tensor,
    relative_samples: Tensor,
) -> Tensor:
    return a[..., None, :] + relative_samples[:, None] * (b - a)[..., None, :]


def sample_along_line_segment(
    a: Tensor,
    b: Tensor,
    n_samples: int,
    *,
    even_sampling: bool = True,
    include_endpoints: bool = True,
    key: 'jax.random.PRNGKey',
) -> Tensor:
    """
    Sample points along a line segment.
    
    Parameters
    ----------
    a : Tensor
        Tensor of shape `(*, p)` containing the coordinates of the start of the
        line segment.
    b : Tensor
        Tensor of shape `(*, p)` containing the coordinates of the end of the
        line segment.
    n_samples : int
        Number of samples to take along the line segment.
    even_sampling : bool (default True)
        Whether to sample evenly along the line segment. If this is False, then
        samples are drawn from a uniform distribution along the line segment.
    include_endpoints : bool (default True)
        Whether to include the endpoints of the line segment in the samples.
    key : jax.random.PRNGKey
        Key to use for random number generation.

    Returns
    -------
    Tensor
        Tensor of shape `(*, n_samples, p)` containing the sampled points.
    """
    rel = _obtain_relative_samples(
        n_samples=n_samples,
        even_sampling=even_sampling,
        include_endpoints=include_endpoints,
        key=key,
    )
    return _obtain_absolute_samples(a, b, rel)


def integrate_along_line_segment(
    a: Tensor,
    b: Tensor,
    metric_tensor_field: Callable[[Tensor], Tensor],
    n_samples: int,
    *,
    norm: callable = quadratic_form,
    even_sampling: bool = True,
    include_endpoints: bool = True,
    weight_by_euc_length: bool = True,
    weight_by_fraction: bool = False,
    key: 'jax.random.PRNGKey',
) -> Tensor:
    """
    Approximately integrate a metric tensor field along a line segment.

    It's generally challenging to find the geodesic distance between two
    points on a Riemannian manifold. Here we use a crude approximation: we
    sample points along a line segment between the two points, and then
    approximate the integral of the metric tensor field along the line segment
    as the average of the metric tensor field at the sampled points.

    Parameters
    ----------
    a : Tensor
        Tensor of shape `(*, p)` containing the coordinates of the start of the
        line segment.
    b : Tensor
        Tensor of shape `(*, p)` containing the coordinates of the end of the
        line segment.
    metric_tensor_field : callable
        Callable that takes a tensor of shape `(*, p)` and returns a tensor of
        shape `(*, p, p)` containing the metric tensor at that point.
    n_samples : int
        Number of samples to take along the line segment.
    norm : callable (default quadratic_form)
        Callable that takes a tensor of shape `(*, p)` and some representation
        of the metric tensor of shape `(*, p, p)` and returns a tensor of
        shape `(*, 1)` containing the norm of the vector under the metric
        tensor. The default is the quadratic form :math:`x^T A x`.

        .. note::
            When approximating the metric tensor field as a sum of a diagonal
            and a low-rank component, the quadratic form can be computed more
            efficiently using :func:`quadratic_form_low_rank_plus_diag`.
    even_sampling : bool (default True)
        Whether to sample evenly along the line segment. If this is False,
        then samples are drawn from a uniform distribution along the line
        segment.
    include_endpoints : bool (default True)
        Whether to include the endpoints of the line segment in the samples.
    weight_by_euc_length : bool (default False)
        Whether to weight the estimated norm of each line segment by its
        Euclidean length.
    weight_by_fraction : bool (default False)
        Whether to weight each sample by the length of the line segment
        between it and the next sample, as a fraction of the total Euclidean
        length of the line segment. This is only relevant if `even_sampling`
        is False.
    key : jax.random.PRNGKey
        Key to use for random number generation.
    """
    rel = _obtain_relative_samples(
        n_samples=n_samples,
        even_sampling=even_sampling,
        include_endpoints=include_endpoints,
        key=key,
    )
    samples = _obtain_absolute_samples(a, b, rel)
    metric_tensor = metric_tensor_field(samples)
    length = norm(samples, metric_tensor)
    if weight_by_fraction and not even_sampling:
        frac = jnp.diff(jnp.sort(rel), prepend=0, append=1)
        weight = ((frac[1:] + frac[:-1]) / 2).sum()
        weight = weight / weight.sum()[..., None]
        length = length * weight
    if weight_by_euc_length:
        euc_length = jnp.linalg.norm(b - a, axis=-1, keepdims=True)[..., None]
        length = length * euc_length
    return length.sum(-2) / n_samples
