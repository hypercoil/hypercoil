# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Synthetic data
~~~~~~~~~~~~~~
Creation process for a small synthetic test dataset.
"""


import numpy as np
import nibabel as nb


def mixing(seed=666, p=50, d=5):
    """
    Create a mixing matrix to produce linear combinations of a low-dimensional
    signal.
    """
    np.random.seed(666) # This hardcoding is intentional.
    mixing = np.random.randint(-10, 10, (p, d)).astype('float')
    mixing[mixing < -3] = 0
    mixing[mixing > 5] = 0
    np.random.seed(seed)
    mixing += np.random.randn(p, d) / 10
    return mixing


def random_ts(seed=666, n=500, d=5):
    """
    Generate a random time series from the Fourier domain, using the specified
    number of channels and observations.
    """
    ts_orig = np.random.randn(n, d)
    fdom = fft.rfft(ts_orig, axis=0)

    nbins = n // 2 + 1
    f = np.linspace(0, 1, nbins)
    ampl = (np.exp(-np.abs(10 * (f - 0.1) ** 2))
            + np.random.randn(d, nbins) / 10)
    ampl[:, 0] = 0
    phase = (np.linspace(-np.pi, np.pi, nbins)
             + np.random.randn(d, nbins) / 10
             + 2 * np.pi * np.random.rand())
    spec = ampl * np.exp(phase * 1j)
    ts = fft.irfft((spec.T * fdom), axis=0)
    return ts


def package_image(seed=666, n=500, d=5, ax=4):
    """
    Generate data and package it into an image.
    """
    p = ax ** 3
    ts = random_ts(seed=seed, n=n, d=d) @ mixing(seed=seed, p=p, d=d).T
    np.random.seed(seed)
    ts += np.random.randn(*ts.shape) / 5
    ts = ts.T.reshape(ax, ax, ax, -1)

    affine = np.eye(4)

    return nb.Nifti1Image(dataobj=ts, affine=affine)
