# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
The von Mises-Fisher distribution on the sphere
"""
from functools import partial
from math import prod
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp

from numpyro.distributions import Distribution, constraints

from ..engine import Tensor, _to_jax_array
from . import NormSphereParameter


#@partial(jax.jit, static_argnames=('shape', 'max_iter'))
def random_VMF(
    mu: Tensor,
    kappa: Tensor,
    shape: Optional[Sequence[int]] = None,
    max_iter: int = 5,
    *,
    key: 'jax.random.PRNGKey',
) -> Tuple[Tensor, Tensor]:
    """
    Von Mises-Fisher distribution sampler with
    mean direction mu and concentration kappa.
    Adapted from source: https://hal.science/hal-04004568
    """
    key_radial, key_angular = jax.random.split(key, 2)

    # parse input parameters
    mu = jnp.atleast_2d(mu)
    batch_size = 1 if shape is None else prod(shape)
    d = mu.shape[-1]
    batch_shape = shape or ()

    # z component: radial samples perpendicular to mu
    z = jax.random.normal(key_radial, (batch_size, *mu.shape))
    z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)
    z = z - jnp.einsum('...kd,kd1->...k1', z, mu[..., None]) * mu
    z = z / jnp.linalg.norm(z, axis=-1, keepdims=True)

    # sample angles (in cos and sin form)
    sin, cos, found = random_VMF_angle(
        d=d,
        k=kappa,
        batch_size=batch_size,
        event_shape=mu.shape[:-1],
        max_iter=max_iter,
        key=key_angular,
    )

    # combine angles with the z component
    x = z * sin[..., None] + cos[..., None] * mu
    return x.reshape((*batch_shape, *mu.shape)), found


def random_VMF_angle(
    d: int,
    k: Tensor,
    batch_size: int,
    event_shape: Sequence[int] = (),
    max_iter: int = 5,
    *,
    key: 'jax.random.PRNGKey',
) -> Tuple[Tensor, Tensor]:
    """
    Generate n iid samples t with density function given by
    p(t) = C * (1 - t ** 2) ** ((d - 3) / 2) * exp(kappa * t)
    for some constant C
    Adapted from source: https://hal.science/hal-04004568
    """
    shape = (batch_size, *event_shape)
    alpha = (d - 1) / 2
    t0 = r0 = jnp.sqrt(1 + (alpha / k) ** 2) - alpha / k
    log_t0 = k * t0 + (d - 1) * jnp.log(1 - r0 * t0)

    found = jnp.zeros(shape, dtype=bool)
    cos = jnp.zeros(shape)
    # TODO: This is slow to compile because the compiler
    # unrolls the loop. See if we can make it faster with jax.lax.scan
    for i in range(max_iter):
        key_i = jax.random.fold_in(key, i)
        key_beta, key_thresh = jax.random.split(key_i)
        t = jax.random.beta(
            key_beta,
            a=alpha,
            b=alpha,
            shape=shape,
        )
        t = 2 * t - 1
        t = (r0 + t) / (1 + r0 * t)
        accept = jnp.exp(
            k * t + (d - 1) * jnp.log(1 - r0 * t) - log_t0
        )
        thresh = jax.random.uniform(key_thresh, shape)
        cos = jnp.where(~found, t, cos)
        found = jnp.where(thresh < accept, True, found)
    sin = jnp.sqrt(1 - cos**2)
    return sin, cos, found


def log_bessel(order: int, kappa: Tensor) -> Tensor:
    """
    Approximation to the logarithm of a modified Bessel function
    of the first kind. We use this to normalise the vMF
    distribution.

    Adapted from source:
    https://github.com/DiedrichsenLab/HierarchBayesParcel/blob/main/emissions.py#L1473

    This looks like it's an asymptotic approximation for large
    order and kappa. We should verify that this is the case and
    that it's accurate for the range of values we expect.
    """
    frac = kappa / order
    square = 1 + frac**2
    root = jnp.sqrt(square)
    eta = root + jnp.log(frac) - jnp.log(1 + root)
    return -jnp.log(
        jnp.sqrt(2 * jnp.pi * order)
    ) + order * eta - 0.25 * jnp.log(square)


def log_prob_vmf(X: Tensor, mu: Tensor, kappa: Tensor) -> Tensor:
    d = X.shape[-1]
    log_norm = (
        (d / 2 - 1) * jnp.log(kappa) -
        (d / 2) * jnp.log(2 * jnp.pi) -
        log_bessel(order=d / 2 - 1, kappa=kappa)
    )
    #return kappa * mu @ X.swapaxes(-2, -1) + log_norm
    return jnp.einsum(
        '...ld,...nd,...l->...nl',
        jnp.atleast_2d(mu),
        X,
        jnp.atleast_1d(kappa),
    ) + log_norm


def vmf_logpdf(mu: Tensor, kappa: Tensor) -> callable:
    def logpdf(X: Tensor) -> Tensor:
        return log_prob_vmf(X=X, mu=mu, kappa=kappa)
    return logpdf


class VonMisesFisher(Distribution):
    support = constraints.sphere
    pytree_aux_fields = ('sample_max_iter', 'sample_return_valid')

    def __init__(
        self,
        mu: Tensor,
        kappa: Tensor,
        sample_max_iter: int = 5,
        sample_return_valid: bool = False,
        explicit_normalisation: bool = True,
    ):
        event_shape = mu.shape[-1]
        batch_shape = mu.shape[:-1]
        self.mu = NormSphereParameter(
            model=mu,
            where=lambda x: x,
            norm=2,
        )
        self.kappa = kappa
        self.sample_max_iter = sample_max_iter
        self.sample_return_valid = sample_return_valid
        self.explicit_normalisation = explicit_normalisation
        super().__init__(
            batch_shape=batch_shape,
            event_shape=event_shape,
        )

    def sample(
        self,
        key: 'jax.random.PRNGKey',
        sample_shape: Sequence[int] = (),
    ) -> Tensor:
        #TODO: See if we can implement reparameterisations so that we can
        # differentiate through this. Low priority since we don't use this
        sample, check = random_VMF(
            mu=_to_jax_array(self.mu),
            kappa=_to_jax_array(self.kappa),
            shape=sample_shape,
            max_iter=self.sample_max_iter,
            key=key,
        )
        if self.sample_return_valid:
            return sample, check
        return sample

    def log_prob(self, value: Tensor) -> Tensor:
        if self.explicit_normalisation:
            value = value / jnp.linalg.norm(value, axis=-1, keepdims=True)
        return vmf_logpdf(
            mu=_to_jax_array(self.mu),
            kappa=_to_jax_array(self.kappa),
        )(value)
