# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for connectopic gradient mapping.
"""
import pytest
import torch
import matplotlib.pyplot as plt
from pkg_resources import resource_filename as pkgrf
from hypercoil.functional.connectopy import (
    laplacian_eigenmaps
)
from brainspace.gradient.embedding import(
    laplacian_eigenmaps as le_ref
)


class TestConnectopy:

    def plot_gradients(self, ref, test, name):
        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].scatter(ref[0], ref[1])
        ax[0].set_title('Reference')
        ax[1].scatter(test[0], test[1])
        ax[1].set_title('Test')
        results = pkgrf(
            'hypercoil',
            'results/'
        )
        fig.savefig(f'{results}/eigenmaps_{name}.png')

    def test_circle_eigenmaps(self):
        dim = 101
        A = torch.diag_embed(torch.ones(dim - 1), 1)
        A[0, -1] = 1
        A = A + A.t()
        L, Q = laplacian_eigenmaps(A)
        Q_ref, L_ref = le_ref(A.numpy())
        self.plot_gradients(
            ref=Q_ref[..., :2].T,
            test=Q[..., :2].t(),
            name='circle'
        )
