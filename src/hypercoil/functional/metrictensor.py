# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Parameterise and integrate over a p-dimensional metric tensor field.

We use these operations to parameterise a latent connectopic space as a
Riemannian manifold. The metric tensor field is a p-dimensional tensor field
that defines the Riemannian metric at each point in the manifold.
"""
from typing import Callable

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
    if even_sampling:
        if include_endpoints:
            gen = jnp.arange(n_samples) / (n_samples - 1)
        else:
            gen = jnp.arange(1, n_samples + 1) / (n_samples + 2)
    else:
        gen = jax.random.uniform(shape=(n_samples,), key=key)
    return a[..., None, :] + gen[:, None] * (b - a)[..., None, :]


def integrate_along_line_segment(
    a: Tensor,
    b: Tensor,
    metric_tensor_field: Callable[[Tensor], Tensor],
    n_samples: int,
    *,
    even_sampling: bool = True,
    include_endpoints: bool = True,
    weight_by_euc_length: bool = False,
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
        Tensor of shape `(p,)` containing the coordinates of the start of the
        line segment.
    b : Tensor
        Tensor of shape `(p,)` containing the coordinates of the end of the
        line segment.
    metric_tensor_field : callable
        Callable that takes a tensor of shape `(p,)` and returns a tensor of
        shape `(p, p)` containing the metric tensor at that point.
    n_samples : int
        Number of samples to take along the line segment.
    even_sampling : bool (default True)
        Whether to sample evenly along the line segment. If this is False, then
        samples are drawn from a uniform distribution along the line segment.
    include_endpoints : bool (default True)
        Whether to include the endpoints of the line segment in the samples.
    weight_by_euc_length : bool (default False)
        Whether to weight each sample by the length of the line segment between
        it and the next sample. This is only relevant if `even_sampling` is
        False.
    key : jax.random.PRNGKey
        Key to use for random number generation.
    """
    samples = sample_along_line_segment(
        a, b, n_samples,
        even_sampling=even_sampling,
        include_endpoints=include_endpoints,
        key=key,
    )
    metric_tensor = metric_tensor_field(samples)
    norm = samples[..., None, :] @ metric_tensor @ samples[..., None]
    norm = norm.squeeze(-1)
    if weight_by_euc_length and not even_sampling:
        euc_length = jnp.linalg.norm(samples, axis=-1, keepdims=True)
        norm = norm * euc_length
    return norm.sum(-2) / n_samples
