# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlas synthesis
~~~~~~~~~~~~~~~
Synthesise some simple parcellated ground truth datasets.
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Any, List, Literal, Optional, Tuple
from hypercoil.engine.paramutil import Tensor
from hypercoil.functional.sphere import euclidean_conv
from .mix import (
    synth_slow_signals,
    mix_data_01,
)


def hard_atlas_example(d: int = 25) -> Tensor:
    """
    Synthesise a simple hard parcellation ground truth. The parameter
    d specifies the dimension of the parcellated image.
    """
    A = jnp.zeros((d, d))
    ax_x = jnp.tile(jnp.arange(d).reshape((1, -1)), (d, 1))
    ax_y = jnp.tile(jnp.arange(d).reshape((-1, 1)), (1, d))

    c = d // 2
    b = d // 3

    R = jnp.sqrt((ax_x - c) ** 2 + (ax_y - c) ** 2)
    A = A.at[ax_x >= c].set(1)
    A = A.at[ax_y >= c].set(2)
    A = A.at[jnp.logical_and(ax_x >= d // 2, ax_y >= d // 2)].set(3)
    A = A.at[R < c].set(4)
    A = A.at[jnp.logical_and(A == 4, ax_x > d // 2)].set(5)
    A = A.at[R < b].set(6)
    A = A.at[jnp.logical_and(A == 6, ax_y > b + 1)].set(7)
    A = A.at[jnp.logical_and(A == 7, ax_y >= (d - b - 2))].set(8)
    return A


def hard_atlas_homologue(d: int = 25) -> Tensor:
    """
    Synthesise an alternative simple hard parcellation ground truth, whose
    parcels exhibit clear homology with those from `hard_atlas_example`. The
    parameter d specifies the dimension of the parcellated image.
    """
    A = jnp.zeros((d, d))
    ax_x = jnp.tile(jnp.arange(d).reshape((1, -1)), (d, 1))
    ax_y = jnp.tile(jnp.arange(d).reshape((-1, 1)), (1, d))

    c = d // 2 + 1
    b = d // 3

    R = jnp.sqrt((ax_x - c) ** 2 + (ax_y - c) ** 2)
    R0 = jnp.sqrt((ax_x - c + 1) ** 2 + (ax_y - c + 1) ** 2)
    A = A.at[(0.8 * ax_x + 0.4 * ax_y) >= c].set(1)
    A = A.at[(0.7 * ax_y - 0.6 * ax_x + 6) >= c].set(2)
    A = A.at[jnp.logical_and(ax_x >= c - 3, ax_y >= c + 3)].set(3)
    A = A.at[R < (c - 2)].set(4)
    A = A.at[jnp.logical_and(A == 4, (0.8 * ax_x - 0.3 * ax_y + 5) > d // 2)].set(5)
    A = A.at[R0 < b].set(6)
    A = A.at[jnp.logical_and(A == 6, (0.9 * ax_y - 0.2 * ax_x + 3) > b + 1)].set(7)
    A = A.at[jnp.logical_and(A == 7, (0.9 * ax_y + 0.2 * ax_x) >= (d - b - 2))].set(8)
    return A


def soft_atlas_example(
    d: int = 25,
    c: int = 9,
    walk_length: int = 15,
    *,
    key: 'jax.random.PRNGKey',
) -> Tensor:
    """
    Synthesise a simple soft parcellation ground truth.

    The parcellation is synthesised as follows:
    1. For each parcel, randomly select an origin point.
    2. Take a random walk from that origin point, adding each traversed point
       to the parcel.
    3. Convolve the parcel using a Gaussian with scale parameter d // 5.
    4. Square and normalise parcels.

    Parameters
    ----------
    d : int (default 25)
        Dimension of the parcellated image.
    c : int (default 9)
        Number of parcels.
    walk_length : int (default 15)
        Length of random walk used when generating parcels.
    seed : int (default None)
        Seed for RNG.
    """
    ax_x = jnp.tile(jnp.arange(d).reshape((1, -1)), (d, 1))
    ax_y = jnp.tile(jnp.arange(d).reshape((-1, 1)), (1, d))

    A = jnp.zeros((c, d, d))
    keys = jax.random.split(key, c)
    for i, k in zip(range(c), keys):
        k_0, k_1, k_2 = jax.random.split(k, 3)
        x_i = jax.random.randint(k_0, minval=0, maxval=d, shape=(1,))
        y_i = jax.random.randint(k_1, minval=0, maxval=d, shape=(1,))
        A = A.at[i, x_i, y_i].set(1)
        j = 0
        while j < walk_length:
            k_2 = jax.random.split(k_2, 1)[0]
            direction = jax.random.randint(
                k_2, minval=0, maxval=4, shape=(1,))
            if direction == 0:
                x_i = min([x_i + 1, d - 1])
            if direction == 1:
                x_i = max([x_i - 1, 0])
            if direction == 2:
                y_i = min([y_i + 1, d - 1])
            if direction == 3:
                y_i = max([y_i - 1, 0])
            A = A.at[i, x_i, y_i].set(1)
            j += 1
            
    coor = jnp.stack((ax_x.reshape((d * d,)), ax_y.reshape((d * d,))))
    A = A.reshape((c, d * d))
    A = euclidean_conv(
        A.swapaxes(-1, -2),
        coor.swapaxes(-1, -2),
        scale=(d / 5),
        max_bin=10000,
        truncate=None
    )
    A = (A ** 2)
    Amax = A.max(1, keepdims=True)
    A = (A / Amax).swapaxes(-1, -2)
    A = A - A.mean()
    return A.reshape((c, d, d))


def hierarchical_atlas_example(
    init: Any = None,
    d: int = 25,
    axis: int = 0,
    sources: Any = None,
    scales: Optional[List] = None,
    divide: int = 5,
    t: int = 300,
    latent_dim: int = 100,
    *,
    key: 'jax.random.PRNGKey',
) -> Tensor:
    """
    Synthesise a hierarchical parcellation ground truth. Unlike the example
    hard and soft parcellations, note that this returns both the time series
    dataset and a representation of the parcellation. Also note that the
    parcellation representation is currently wrong but nonetheless
    intelligible.

    A hierarchical parcellation proceeds by partitioning space into `divide`
    subparts along the specified `axis`. It then repeats this process,
    switching axes at each depth. A time series particular to each cell is
    added in proportion to the `scale` of that tier of the hierarchy.
    (The first tier corresponds to a global signal.) The same sources are
    sampled from at all tiers.

    Parameters
    ----------
    init : Tensor or None (default None)
        Used by the recursive call. Leave as None.
    d : int (default 25)
        Dimension of the parcellated image.
    axis : 0 or 1 (default 0)
        Axis along which the highest level of partition proceeds.
    sources : Tensor or None (default None)
        Used by the recursive call. Leave as None.
    seed : int (default None)
        Seed for RNG.
    scales : list (default [0.1, 0.5, 0.5, 0.2, 0.1])
        Scales applied to time series at each level of the hierarchy.
    divide : int
        Number of subsections to partition into at each level of the
        hierarchy.
    t : int (default 300)
        Time dimension of time series data.
    latent_dim : int (default 100)
        Number of latent signals to generate and embed.
    """
    key_s, key_m, key_r = jax.random.split(key, 3)
    if init is None: init = jnp.ones((d, d))
    if scales is None: scales = [0.1, 0.5, 0.5, 0.2, 0.1]
    if sources is None:
        sources = synth_slow_signals(
            time_dim=t,
            signal_dim=latent_dim,
            key=key_s,
        )

    mix = scales[0] * mix_data_01(sources, mixture_dim=1, key=key_m)
    ts = init.reshape((d, d, 1)) * mix.reshape((1, 1, -1))
    parc = init
    scales = scales[1:]

    if len(scales) > 0:
        S = (init.sum(int(not axis)) != 0)
        loc = jnp.where(S)
        partition_min = loc[0].min().item()
        partition_max = loc[0].max().item()
        partition_step = int((partition_max + 1 - partition_min) / divide)
        start = partition_min
        for i in range(divide):
            key_r = jax.random.split(key_r, 1)[0]
            new_init = jnp.zeros((d, d))
            end = start + partition_step
            if axis == 0:
                new_init = new_init.at[start:end, :].set(1)
            elif axis == 1:
                new_init = new_init.at[:, start:end].set(1)
            new_init = init * new_init
            ts_loc, parc_loc = hierarchical_atlas_example(
                init=new_init,
                axis=int(not axis),
                sources=sources,
                scales=scales,
                divide=divide,
                key=key_r,
            )
            parc = parc + (i + 1) * parc_loc
            ts = ts + ts_loc
            start = end
        axis = int(not axis)
    return ts, parc


def plot_hierarchical(
    parc: Tensor,
    save: Optional[str] = None,
):
    """
    Plot a representation of a hierarchical atlas. This is a dumb function and
    shouldn't be used outside of the specific experiment where it's called.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(jnp.log(parc), cmap='magma')
    plt.xticks([])
    plt.yticks([])
    if save is not None:
        plt.savefig(save, bbox_inches='tight')


def plot_atlas(
    parcels: Tensor,
    d: int,
    c: int = 9,
    saveh: Optional[str] = None,
    saves: Optional[str] = None,
) -> None:
    """
    Plotting utility for synthetic atlases, or for learned models of them.

    Parameters
    ----------
    parcels : tensor
        Tensor containing each voxel's affiliation weight to each parcel.
        Its dimension should be the number of parcels by the number of
        voxels.
    d : int
        Dimension of parcellated image.
    saveh : str or None
        Location to save the figure representing the hard view of the parcels.
    saves : str or None
        Location to save the figure representing the soft view of the parcels.
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(
        parcels.argmax(0).reshape((d, d)),
        cmap='magma',
        vmin=0,
        vmax=parcels.shape[0]
    )
    plt.xticks([])
    plt.yticks([])
    if saveh:
        plt.savefig(saveh, bbox_inches='tight')
    plt.figure(figsize=(8, 8))
    q = int(jnp.ceil(jnp.sqrt(c)))
    for i in range(c):
        plt.subplot(q, q, i + 1)
        lim = jnp.abs(parcels[i, :]).max()
        plt.imshow(
            parcels[i, :].reshape((d, d)),
            vmin=-lim,
            vmax=lim,
            cmap='coolwarm'
        )
        plt.xticks([])
        plt.yticks([])
    if saves:
        plt.savefig(saves, bbox_inches='tight')


def embed_data_in_atlas(
    A: Tensor,
    t: int = 300,
    signal_dim: int = 100,
    atlas_dim: int = 9,
    image_dim: int = 25,
    lp: float = 0.3,
    parc: Literal['soft', 'hard'] = 'hard',
    ts: Optional[Tensor] = None,
    ts_reg: Optional[Tensor] = None,
    *,
    key: Optional['jax.random.PRNGKey'] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Embed the specified time series in the specified atlas.

    Parameters
    ----------
    A : np array
        Atlas array.
    t : int (default 300)
        Time dimension of time series data.
    signal_dim : int (default 300)
        Number of latent signals to generate and embed.
    atlas_dim : int (default 9)
        Number of atlas parcels. Also the number of observed signals to
        compose as linear combinations of the latent signals.
    image_dim : int (default 25)
        Image dimension (side length in pixels) of the parcellation.
    lp : float (default 0.3)
        Fraction of lowest frequencies to spare from obliteration.
    parc : 'soft' or 'hard'
        Kind of parcellation.
    ts : array
        If specified, use these as the latent time series.
    ts_reg : array
        If specified, use these as the observed regional time series. Note
        that the argument `ts` is ignored if this is specified.
    """
    if ts_reg is None:
        key_0, key_1 = jax.random.split(key)
        if ts is None:
            ts = synth_slow_signals(
                time_dim=t, signal_dim=signal_dim, lp=lp, key=key_0,
            )
        ts_reg = mix_data_01(
            ts=ts, mixture_dim=atlas_dim, key=key_1,
        )
    if parc == 'hard':
        data = jnp.zeros((image_dim, image_dim, t))
        for i in range(atlas_dim):
            data = data.at[A == i].set(ts_reg[i, :].reshape(1, -1))
    elif parc == 'soft':
        data = (
            A.reshape((atlas_dim, image_dim * image_dim)).swapaxes(-1, -2) @
            ts_reg
        )
        data = data.reshape((image_dim, image_dim, t))
    return data, ts_reg


def get_model_matrices(
    A: Tensor,
    data: Tensor,
    d: int = 25,
    r: int = 9,
    t: int = 300,
    parc: Literal['hard', 'soft'] = 'hard',
) -> Tuple[Tensor, Tensor]:
    """
    Obtain the atlas and the time series in matrix form for easy use in
    training.

    Parameters
    ----------
    A : array
        Atlas as a spatial array.
    data : tensor
        Time series as a spatial array.
    d : int (default 25)
        Image dimension (side length in pixels) of the parcellation.
    r : int (default 9)
        Number of atlas parcels.
    t : int (default 300)
        Time dimension of time series data.
    parc : 'soft' or 'hard'
        Kind of parcellation.

    Returns
    -------
    ref : tensor
        Atlas as a regions x pixels tensor.
    tsmat : tensor
        Time series as a pixels x time tensor.
    """
    tsmat = data.reshape((d * d, t))
    if parc == 'hard':
        ref = jnp.zeros((r, d * d))
        Avec = A.ravel()
        for i in range(r):
            ref = ref.at[i, :].set((Avec == i).astype(float))
    elif parc == 'soft':
        ref = A.reshape((r, d * d))
    return ref, tsmat
