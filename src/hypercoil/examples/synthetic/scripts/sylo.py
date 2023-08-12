# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data synthesis: sylo
~~~~~~~~~~~~~~~~~~~~
Synthesise data experimentally testing the sylo module.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt
from functools import partial
from typing import Callable, Optional
from hypercoil.engine.paramutil import PyTree, Tensor
from hypercoil.nn import Sylo, Recombinator


def synthesise_lowrank_block(
    H: Tensor,
    *,
    key: 'jax.random.PRNGKey',
):
    # This noise level seems to work all right once you get to
    # around 50 nodes. Not much rationale as this function is
    # a one-off
    noise_level = .5 / jnp.log(H)
    
    data_low = jnp.zeros([H, 4])
    data_low = data_low.at[:(H // 4), 0].set(0.6)
    data_low = data_low.at[(H // 4):(3 * H // 5), 0].set(-0.3)
    data_low = data_low.at[(4 * H // 5):, 0].set(-0.6)
    data_low = data_low.at[(H // 4):(2 * H // 3), 1].set(0.4)
    data_low = data_low.at[(2 * H // 3):, 1].set(-0.3)
    data_low = data_low.at[(H // 2):(4 * H // 5), 2].set(0.5)
    data_low = data_low.at[(H // 5):(2 * H // 5), 2].set(0.4)
    data_low = data_low.at[:(H // 9), 3].set(0.3)
    data_low = data_low.at[(H // 9):(H // 3), 3].set(-0.5)
    data_low = data_low.at[(7 * H // 9):(8 * H // 9), 3].set(0.6)

    noise = noise_level * jax.random.normal(key=key, shape=(H, H))

    A = jnp.tanh(data_low @ data_low.T) + noise @ noise.T
    return A


def plot_templates(
    model: PyTree,
    X: Tensor,
    n_filters: int,
    save: Optional[str] = None,
):
    plt.figure(figsize=(5 * n_filters, 10))
    out = model(X, key=jax.random.PRNGKey(0)).squeeze()
    wei = model.weight[0].squeeze()
    for i in range(n_filters):
        plt.subplot(2, n_filters, i + 1)
        lim = jnp.quantile(jnp.abs(out[i]), .95)
        plt.imshow(out[i], cmap='RdBu_r', vmin=-lim, vmax=lim)
        t = wei[i].reshape(-1, 1) @ wei[i].reshape(1, -1)
        plt.subplot(2, n_filters, i + 1 + n_filters)
        lim = jnp.quantile(jnp.abs(t), .95)
        plt.imshow(t, cmap='RdBu_r', vmin=-lim, vmax=lim)
        plt.xticks([])
        plt.yticks([])
    if save:
        plt.savefig(save)


def plot_outcome(
    model: PyTree,
    X: Tensor,
    save: Optional[str] = None,
):
    out = model(X, key=jax.random.PRNGKey(0)).squeeze()
    plot_conn(out, save=save)


def plot_conn(
    conn: Tensor,
    save: Optional[str] = None,
):
    plt.figure(figsize=(10, 10))
    lim = jnp.quantile(jnp.abs(conn), .95)
    plt.imshow(conn, cmap='RdBu_r', vmin=-lim, vmax=lim)
    plt.xticks([])
    plt.yticks([])
    if save:
        plt.savefig(save)


class SyloShallowAutoencoder(eqx.Module):
    dim: int
    sylo: PyTree
    nlin: Callable
    rcmb: PyTree

    def __init__(
        self,
        n_channels: int,
        n_filters: int,
        dim: int,
        mix_bias: bool = True,
        rank: int = 1,
        *,
        key: 'jax.random.PRNGKey',
    ):
        key_s, key_r = jax.random.split(key)
        self.dim = dim
        self.sylo = Sylo(
            in_channels=n_channels,
            out_channels=n_filters,
            dim=dim,
            rank=rank,
            bias=True,
            remove_diagonal=True,
            key=key_s,
        )
        self.nlin = partial(jax.nn.leaky_relu, negative_slope=0.2)
        self.rcmb = Recombinator(
            in_channels=n_filters,
            out_channels=n_channels,
            bias=mix_bias,
            positive_only=True,
            key=key_r,
        )

    def __call__(self, X, *, key: 'jax.random.PRNGKey'):
        out = self.sylo(X)
        out = self.nlin(out)
        out = self.rcmb(out)
        return out.squeeze()
