# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Synthetic data
~~~~~~~~~~~~~~
Creation process for a small synthetic test dataset.
This code is intended as a one-off for helping with some unit tests. It's
not intended to be generalisable, reusable, or applicable outside of its
original context.
"""
import json
import numpy as np
import nibabel as nb
import pandas as pd
from scipy import fft
from scipy.signal import convolve
from itertools import product
from collections import OrderedDict


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
    np.random.seed(seed)
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


def package_image(seed=666, n=500, d=5, ax=4, region=None, pattern=None):
    """
    Generate data and package it into an image.
    """
    p = ax ** 3
    ts = random_ts(seed=seed, n=n, d=d) @ mixing(seed=seed, p=p, d=d).T
    np.random.seed(seed)
    ts += np.random.randn(*ts.shape) / 5
    ts = ts.T.reshape(ax, ax, ax, -1)

    ts += add_stim(seed=seed, ax=ax, region=region, pattern=pattern)

    affine = np.eye(4)

    return nb.Nifti1Image(dataobj=ts, affine=affine)


def stim_pattern(on=10, off=40, n=500):
    # The generation process is pretty meaningless and extremely crude.
    # Please do not consider it for any actual scientific purpose.
    cycle = np.zeros(on + off)
    cycle[-on:] = 1
    pattern = np.tile(cycle, n // (on + off) + 1)

    xs = np.arange(1, (on + off) + 1)
    z = [3.5, 2]
    f1 = xs ** z[0] * np.exp(-xs)
    f2 = (xs / 3) ** z[1] * np.exp(-(xs / 3))
    pattern = convolve(pattern, f1 - f2, mode='same')

    return pattern[:n]


def stim_roi(ax=4, scale=1):
    roi = np.zeros((ax, ax, ax, 1))
    roi[:2, 1:3, 1:3] = scale
    return roi


def add_stim(seed=666, ax=4, region=stim_roi, pattern=stim_pattern):
    if pattern is not None:
        pattern = pattern()
        if region is not None:
            region = region()
        else:
            region = np.ones((ax, ax, ax, 1))
        region[region!=0] += (
            np.random.randn(*region[region!=0].shape) / 5)
        return region * pattern
    else:
        return np.zeros((ax, ax, ax, 1))


def regressor_ts(img, seed=666, n=500):
    confs = {}
    np.random.seed(seed)

    # fake mean signals
    img = img.get_fdata()
    x, y, z, _ = img.shape
    confs['global_signal'] = img.mean((0, 1, 2))
    noise_roi_1 = np.zeros((x, y, z)).astype('bool')
    noise_roi_1[(x // 2):(x // 2 + 1), 1:(y - 1), 1:(z - 1)] = 1
    confs['white_matter'] = img[noise_roi_1].mean(0)
    noise_roi_2 = np.zeros((x, y, z)).astype('bool')
    noise_roi_2[(x // 2 + 1):x, 1:(y - 1), 1:(z - 1)] = 1
    confs['csf'] = img[noise_roi_2].mean(0)

    # fake RPs
    for v in ('x', 'y', 'z'):
        deriv = np.random.laplace(size=n, scale=0.1)
        confs[f'trans_{v}'] = np.cumsum(deriv)
        deriv = np.random.laplace(size=n, scale=0.1)
        confs[f'rot_{v}'] = np.cumsum(deriv)

    metadata = {}

    # fake aCompCor
    k = 0
    for mask in ('CSF', 'WM', 'combined'):
        n_components = np.random.randint(1, 4)
        sigma = 10 * np.random.rand() * (
            1 - np.sort(np.random.rand(n_components)))
        var_exp = sigma ** 2
        var_exp /= var_exp.sum()
        c_var_exp = np.cumsum(var_exp)
        for i in range(n_components):
            confs[f'a_comp_cor_{(k + i):02}'] = random_ts(
                seed=None, n=n, d=1).squeeze()
            metadata[f'a_comp_cor_{(k + i):02}'] = {
                'CumulativeVarianceExplained': c_var_exp[i],
                'Mask': mask,
                'Method': 'aCompCor',
                'Retained': 'True',
                'SingularValue': sigma[i],
                'VarianceExplained': var_exp[i]
            }
        k += n_components

    # fake tCompCor
    n_components = np.random.randint(1, 4)
    sigma = 10 * np.random.rand() * (
        1 - np.sort(np.random.rand(n_components)))
    var_exp = sigma ** 2
    var_exp /= var_exp.sum()
    c_var_exp = np.cumsum(var_exp)
    for i in range(n_components):
        confs[f't_comp_cor_{i:02}'] = random_ts(seed=None, n=n, d=1).squeeze()
        metadata[f't_comp_cor_{i:02}'] = {
            'CumulativeVarianceExplained': c_var_exp[i],
            'Method': 'tCompCor',
            'Retained': 'True',
            'SingularValue': sigma[i],
            'VarianceExplained': var_exp[i]
        }

    # fake AROMA
    n_components = np.random.randint(1, 64)
    for i in range(n_components):
        confs[f'aroma_motion_{i:02}'] = random_ts(
            seed=None, n=n, d=1).squeeze()
        metadata[f'aroma_motion_{i:02}'] = {
            'MotionNoise': True
        }

    return pd.DataFrame(confs), metadata


def synthesise_dataset(root, seed=666, sub=10, ses=0, run=4,
                       task=None, n=500, d=5, ax=4):
    task = task or {'rest': None, 'stim': (stim_pattern, stim_roi)}
    ids = OrderedDict()
    base = []
    for var, val in [('sub', list(range(sub))),
                     ('ses', list(range(ses))),
                     ('run', list(range(run))),
                     ('task', list(task.keys()))]:
        if len(val) > 0:
            ids[var] = val
            base += [f'{var}' + '-{' + f'{var}' + '}']
        else:
            ids[var] = [None]
    base = '_'.join(base)
    out = f'{root}/{base}'
    combinations = list(product(*ids.values()))
    for su, se, ru, ta in combinations:
        seed_cur = seed
        for na, i in (('sub', su), ('ses', se), ('run', ru), ('task', ta)):
            i = i or 0
            try:
                seed_cur *= (abs(i) + 1)
            except TypeError:
                i = ids[na].index(i)
                seed_cur *= (abs(i) + 1)
        seed_cur *= np.random.randint(0, 99999999)
        seed_cur %= (2 ** 30)
        name = out.format(sub=su, ses=se, run=ru, task=ta)
        pattern, region = task[ta] or (None, None)
        img = package_image(
            seed=seed_cur, n=n, d=d, ax=ax,
            region=region, pattern=pattern)
        img_meta = {
            'RepetitionTime': 1.0,
            'SkullStripped': False,
            'TaskName': ta
        }
        confs, conf_meta = regressor_ts(img=img, seed=seed_cur, n=n)
        print(f'Saving {name}')
        nb.save(img, f'{name}_desc-preproc_bold.nii.gz')
        confs.to_csv(f'{name}_desc-confounds_timeseries.tsv',
                     sep='\t', index=False)
        with open(f'{name}_desc-confounds_timeseries.json', 'w') as f:
            json.dump(conf_meta, f)
        with open(f'{name}_desc-preproc_bold.json', 'w') as f:
            json.dump(img_meta, f)
