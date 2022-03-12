# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlas synthesis
~~~~~~~~~~~~~~~
Synthesise some simple parcellated ground truth datasets.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from ..functional.sphere import euclidean_conv
from .mix import (
    synth_slow_signals,
    mix_data_01
)


def hard_atlas_example(d=25):
    """
    Synthesise a simple hard parcellation ground truth. The parameter
    d specifies the dimension of the parcellated image.
    """
    A = torch.zeros((d, d))
    ax_x = torch.arange(d).view(1, -1).tile(d, 1)
    ax_y = torch.arange(d).view(-1, 1).tile(1, d)

    c = d // 2
    b = d // 3

    R = torch.sqrt((ax_x - c) ** 2 + (ax_y - c) ** 2)
    A[ax_x >= c] = 1
    A[ax_y >= c] = 2
    A[torch.logical_and(ax_x >= d // 2, ax_y >= d // 2)] = 3
    A[R < c] = 4
    A[torch.logical_and(A == 4, ax_x > d // 2)] = 5
    A[R < b] = 6
    A[torch.logical_and(A == 6, ax_y > b + 1)] = 7
    A[torch.logical_and(A == 7, ax_y >= (d - b - 2))] = 8
    return A


def hard_atlas_homologue(d=25):
    """
    Synthesise an alternative simple hard parcellation ground truth, whose
    parcels exhibit clear homology with those from `hard_atlas_example`. The
    parameter d specifies the dimension of the parcellated image.
    """
    A = torch.zeros((d, d))
    ax_x = torch.arange(d).view(1, -1).tile(d, 1)
    ax_y = torch.arange(d).view(-1, 1).tile(1, d)

    c = d // 2 + 1
    b = d // 3

    R = torch.sqrt((ax_x - c) ** 2 + (ax_y - c) ** 2)
    R0 = torch.sqrt((ax_x - c + 1) ** 2 + (ax_y - c + 1) ** 2)
    A[(0.8 * ax_x + 0.4 * ax_y) >= c] = 1
    A[(0.7 * ax_y - 0.6 * ax_x + 6) >= c] = 2
    A[torch.logical_and(ax_x >= c - 3, ax_y >= c + 3)] = 3
    A[R < (c - 2)] = 4
    A[torch.logical_and(A == 4, (0.8 * ax_x - 0.3 * ax_y + 5) > d // 2)] = 5
    A[R0 < b] = 6
    A[torch.logical_and(A == 6, (0.9 * ax_y - 0.2 * ax_x + 3) > b + 1)] = 7
    A[torch.logical_and(A == 7, (0.9 * ax_y + 0.2 * ax_x) >= (d - b - 2))] = 8
    return A


def soft_atlas_example(d=25, c=9, walk_length=15, seed=None):
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
    ax_x = torch.arange(d).view(1, -1).tile(d, 1)
    ax_y = torch.arange(d).view(-1, 1).tile(1, d)

    A = torch.zeros((c, d, d))
    np.random.seed(seed)
    for i in range(c):
        x_i = np.random.randint(0, d)
        y_i = np.random.randint(0, d)
        A[i, x_i, y_i] = 1
        j = 0
        while j < walk_length:
            direction = np.random.randint(0, 4)
            if direction == 0:
                x_i = min([x_i + 1, d - 1])
            if direction == 1:
                x_i = max([x_i - 1, 0])
            if direction == 2:
                y_i = min([y_i + 1, d - 1])
            if direction == 3:
                y_i = max([y_i - 1, 0])
            A[i, x_i, y_i] = 1
            j += 1
            
    coor = torch.stack((ax_x.view(d * d), ax_y.view(d * d)))
    A = A.view(c, d * d)
    A = euclidean_conv(
        A.transpose(-1, -2),
        coor.transpose(-1, -2),
        scale=(d / 5),
        max_bin=10000,
        truncate=None
    )
    A = (A ** 2)
    Amax, _ = A.max(1, keepdim=True)
    A = (A / Amax).transpose(-1, -2)
    A = A - A.mean()
    return A.view(c, d, d)


def hierarchical_atlas_example(
    init=None, d=25, axis=0, sources=None, seed=None,
    scales=None, divide=5, t=300, latent_dim=100
):
    if init is None: init = torch.ones((d, d))
    if scales is None: scales = [0.1, 0.5, 0.5, 0.2, 0.1]
    if sources is None:
        sources = synth_slow_signals(
            time_dim=t,
            signal_dim=latent_dim,
            seed=seed
        )

    mix = scales[0] * torch.FloatTensor(mix_data_01(sources, mixture_dim=1))
    ts = init.view(d, d, 1) * mix.view(1, 1, -1)
    parc = init
    #print(ts.shape)
    scales = scales[1:]

    if len(scales) > 0:
        S = (init.sum(int(not axis)) != 0)
        loc = torch.where(S)
        partition_min = loc[0].min().item()
        partition_max = loc[0].max().item()
        partition_step = int((partition_max + 1 - partition_min) / divide)
        #print(partition_min, partition_max, partition_step, scales)
        start = partition_min
        for i in range(divide):
            new_init = torch.zeros((d, d))
            end = start + partition_step
            #print(start, end)
            if axis == 0:
                new_init[start:end, :] = 1
            elif axis == 1:
                new_init[:, start:end] = 1
            new_init = init * new_init
            #print(new_init)
            #plt.figure()
            #plt.imshow(new_init.numpy(), cmap='bone')
            ts_loc, parc_loc = hierarchical_atlas_example(
                init=new_init,
                axis=int(not axis),
                sources=sources,
                scales=scales,
                divide=divide
            )
            parc = parc + (i + 1) * parc_loc
            ts = ts + ts_loc
            start = end
        axis = int(not axis)
    return ts, parc


def plot_hierarchical(parc, save=None):
    plt.figure(figsize=(8, 8))
    plt.imshow(torch.log(parc), cmap='magma')
    plt.xticks([])
    plt.yticks([])
    if save is not None:
        plt.savefig(save)


def plot_atlas(parcels, d, c=9, saveh=None, saves=None):
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
        parcels.argmax(0).view(d, d).detach().numpy(),
        cmap='magma',
        vmin=0,
        vmax=parcels.shape[0]
    )
    plt.xticks([])
    plt.yticks([])
    if saveh:
        plt.savefig(saveh)
    plt.figure(figsize=(8, 8))
    q = int(np.ceil(np.sqrt(c)))
    for i in range(c):
        plt.subplot(q, q, i + 1)
        lim = torch.abs(parcels[i, :]).max().detach().numpy()
        plt.imshow(
            parcels[i, :].view(d, d).detach().numpy(),
            vmin=-lim,
            vmax=lim,
            cmap='coolwarm'
        )
        plt.xticks([])
        plt.yticks([])
    if saves:
        plt.savefig(saves)


def embed_data_in_atlas(A, t=300, signal_dim=100, atlas_dim=9,
                        image_dim=25, lp=0.3, parc='hard',
                        ts=None, ts_reg=None):
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
        if ts is None:
            ts = synth_slow_signals(
                time_dim=t, signal_dim=signal_dim, lp=lp
            )
        ts_reg = mix_data_01(
            ts=ts, mixture_dim=atlas_dim
        )
    if parc == 'hard':
        data = torch.zeros(image_dim, image_dim, t)
        for i in range(atlas_dim):
            data[A == i] = torch.FloatTensor(ts_reg[i, :]).reshape(1, 1, -1)
    elif parc == 'soft':
        data = (
            A.view(atlas_dim, image_dim * image_dim).transpose(-1, -2) @
            torch.FloatTensor(ts_reg)
        )
        data = data.view(image_dim, image_dim, t)
    return data, ts_reg


def get_model_matrices(A, data, d=25, r=9, t=300, parc='hard'):
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
    tsmat = data.view(d * d, t)
    if parc == 'hard':
        ref = torch.zeros(r, d * d)
        Avec = A.ravel()
        for i in range(r):
            ref[i, :] = (Avec == i).float()
    elif parc == 'soft':
        ref = A.view(r, d * d)
    return ref, tsmat
