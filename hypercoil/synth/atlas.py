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


def plot_atlas(parcels, d, saveh=None, saves=None):
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
    for i in range(9):
        plt.subplot(3, 3, i + 1)
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
    if ts_reg is None:
        if ts is None:
            ts = synth_slow_signals(
                time_dim=t, signal_dim=signal_dim, lp=lp
            )
        ts_reg = mix_data(
            ts=ts, atlas_dim=atlas_dim,
            signal_dim=signal_dim
        )
    if parc == 'hard':
        data = torch.zeros(image_dim, image_dim, t)
        for i in range(atlas_dim):
            data[A == i] = torch.FloatTensor(ts_reg[:, i]).reshape(1, 1, -1)
    elif parc == 'soft':
        data = (
            A.view(atlas_dim, image_dim * image_dim).transpose(-1, -2) @
            torch.FloatTensor(ts_reg.T)
        )
        data = data.view(image_dim, image_dim, t)
    return data, ts_reg


def get_model_matrices(A, data, d=25, r=9, t=300, parc='hard'):
    tsmat = data.view(d * d, t)
    if parc == 'hard':
        ref = torch.zeros(r, d * d)
        Avec = A.ravel()
        for i in range(r):
            ref[i, :] = (Avec == i).float()
    elif parc == 'soft':
        ref = A.view(r, d * d)
    return ref, tsmat
