# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Butterworth initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~~
Initialise parameters to approximate the attenuation curve of a Butterworth
filter.
"""
import torch
import math


def butterworth_init_(tensor, N, Wn, btype='bandpass', fs=None):
    if fs is not None:
        Wn = 2 * Wn / fs
    frequencies = torch.linspace(0, 1, tensor.size(-1))
    if btype == 'bandpass':
        hi, lo = Wn
        lovals = 1 / torch.sqrt(1 + (frequencies / lo) ** (2 * N))
        hivals = 1 / torch.sqrt(1 + (hi / frequencies) ** (2 * N))
        vals = lovals * hivals
        vals /= vals.max()
    elif btype == 'lowpass':
        lo = Wn
        vals = 1 / torch.sqrt(1 + (frequencies / lo) ** (2 * N))
    elif btype == 'highpass':
        hi = Wn
        vals = 1 / torch.sqrt(1 + (hi / frequencies) ** (2 * N))
    tensor[:] = vals
