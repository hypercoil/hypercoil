# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data synthesis: sylo
~~~~~~~~~~~~~~~~~~~~
Synthesise data experimentally testing the sylo module.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from hypercoil.nn import Sylo, Recombinator


def synthesise_lowrank_block(H, seed=None):
    np.random.seed(seed)
    # This noise level seems to work all right once you get to
    # around 50 nodes. Not much rationale as this function is
    # a one-off
    noise_level = .5 / np.log(H)
    
    data_low = np.zeros([H, 4])
    data_low[:(H // 4), 0] = 0.6
    data_low[(H // 4):(3 * H // 5), 0] = -0.3
    data_low[(4 * H // 5):, 0] = -0.6
    data_low[(H // 4):(2 * H // 3), 1] = 0.4
    data_low[(2 * H // 3):, 1] = -0.3
    data_low[(H // 2):(4 * H // 5), 2] = 0.5
    data_low[(H // 5):(2 * H // 5), 2] = 0.4
    data_low[:(H // 9), 3] = 0.3
    data_low[(H // 9):(H // 3), 3] = -0.5
    data_low[(7 * H // 9):(8 * H // 9), 3] = 0.6

    noise = noise_level * np.random.randn(H, H)

    A = np.tanh(data_low @ data_low.T) + noise @ noise.T
    return A


def plot_templates(layer, X, n_filters, save=None):
    plt.figure(figsize=(5 * n_filters, 10))
    out = layer(X).detach().squeeze()
    wei = layer.weight_L.squeeze().detach()
    for i in range(n_filters):
        plt.subplot(2, n_filters, i + 1)
        lim = np.quantile(np.abs(out[i]), .95)
        plt.imshow(out[i], cmap='coolwarm', vmin=-lim, vmax=lim)
        t = wei[i].reshape(-1, 1) @ wei[i].reshape(1, -1)
        plt.subplot(2, n_filters, i + 1 + n_filters)
        lim = np.quantile(np.abs(t), .95)
        plt.imshow(t, cmap='coolwarm', vmin=-lim, vmax=lim)
        plt.xticks([])
        plt.yticks([])
    if save:
        plt.savefig(save)


def plot_outcome(model, X, save=None):
    out = model(X).detach().numpy().squeeze()
    plot_conn(out, save=save)


def plot_conn(conn, save=None):
    plt.figure(figsize=(10, 10))
    lim = np.quantile(np.abs(conn), .95)
    plt.imshow(conn, cmap='coolwarm', vmin=-lim, vmax=lim)
    plt.xticks([])
    plt.yticks([])
    if save:
        plt.savefig(save)


class SyloShallowAutoencoder(nn.Module):
    def __init__(self, n_channels, n_filters, dim,
                 mix_bias=True, rank=1):
        super().__init__()
        self.dim = dim
        self.net = nn.Sequential(
            Sylo(in_channels=n_channels,
                 out_channels=n_filters,
                 dim=dim,
                 rank=rank,
                 bias=True,
                 delete_diagonal=True),
            nn.LeakyReLU(negative_slope=0.2),
            Recombinator(in_channels=n_filters,
                         out_channels=n_channels,
                         bias=mix_bias,
                         positive_only=True)
        )

    def forward(self, X):
        out = self.net(X)
        out = out.squeeze()
        return out
