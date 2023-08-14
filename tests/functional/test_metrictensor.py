# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for parameterisation of and approximate integration over metric
tensor fields.
"""
from functools import partial

import jax
import jax.numpy as jnp

from hypercoil.functional import (
    metric_tensor_field_diag_plus_low_rank,
    sample_along_line_segment,
    integrate_along_line_segment,
    # integrate_along_line_segment_adaptive,
    # integrate_along_line_segment_adaptive_trapezoid,
    # integrate_along_line_segment_adaptive_simpson,
    # integrate_along_line_segment_adaptive_gauss_kronrod,
)


def test_metric_tensor_field_diag_plus_low_rank():
    key = jax.random.PRNGKey(0)
    coor = jax.random.uniform(key, (3, 10, 7))

    def low_rank_model():
        weight = jax.random.uniform(shape=(14, 7), key=key)

        def forward(x):
            batch_shape = x.shape[:-1]
            return (weight @ x[..., None]).reshape(batch_shape + (7, 2))

        return forward

    def diag_model():
        def forward(_):
            return jnp.ones((7,))

        return forward

    low_rank_model_ = low_rank_model()
    assert low_rank_model_(coor).shape == (3, 10, 7, 2)
    diag_model_ = diag_model()

    tensors = metric_tensor_field_diag_plus_low_rank(
        coor,
        low_rank_model_,
        diag_model_,
    )
    assert tensors.shape == (3, 10, 7, 7)


def test_sample_along_line_segment():
    key = jax.random.PRNGKey(0)
    key_a, key_b, key_s = jax.random.split(key, 3)
    a = jax.random.uniform(key_a, (3, 7))
    b = jax.random.uniform(key_b, (3, 7))
    n_samples = 10

    samples = sample_along_line_segment(a, b, n_samples, key=key_s)
    assert samples.shape == (3, 10, 7)
    assert jnp.allclose(jnp.diff(samples, 2, axis=-2), 0, atol=1e-6)
    assert jnp.all(samples[:, 0, :] == a)
    assert jnp.all(samples[:, -1, :] == b)

    samples = sample_along_line_segment(
        a, b, n_samples, include_endpoints=False, key=key_s
    )
    assert samples.shape == (3, 10, 7)
    assert jnp.allclose(jnp.diff(samples, 2, axis=-2), 0, atol=1e-6)
    assert jnp.all(samples[:, 0, :] != a)
    assert jnp.all(samples[:, -1, :] != b)

    samples = sample_along_line_segment(
        a, b, n_samples, even_sampling=False, key=key_s
    )
    assert samples.shape == (3, 10, 7)


def test_integrate_along_line_segment():
    key = jax.random.PRNGKey(0)
    key_a, key_b, key_s, key_m = jax.random.split(key, 4)
    a = jax.random.uniform(key_a, (3, 7))
    b = jax.random.uniform(key_b, (3, 7))

    def low_rank_model():
        weight = jax.random.uniform(shape=(14, 7), key=key_m)

        def forward(x):
            batch_shape = x.shape[:-1]
            return (weight @ x[..., None]).reshape(batch_shape + (7, 2))

        return forward

    def diag_model():
        def forward(_):
            return jnp.ones((7,))

        return forward
    
    metric_tensor_field = partial(
        metric_tensor_field_diag_plus_low_rank,
        low_rank_model=low_rank_model(),
        diag_model=diag_model(),
    )

    integral = integrate_along_line_segment(
        a=a,
        b=b,
        metric_tensor_field=metric_tensor_field,
        n_samples=10,
        even_sampling=True,
        include_endpoints=True,
        key=key_s,
    )
    assert integral.shape == (3, 1)
    assert jnp.all(integral >= 0) # positive-definite

    integral_w = integrate_along_line_segment(
        a=a,
        b=b,
        metric_tensor_field=metric_tensor_field,
        n_samples=10,
        even_sampling=False,
        weight_by_euc_length=True,
        key=key_s,
    )
    assert integral_w.shape == (3, 1)
    assert jnp.all(integral_w >= 0) # positive-definite
    assert 0
