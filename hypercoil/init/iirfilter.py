# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
~~~~~~~~THE  REAL~~~~~~~~
IIR filter initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~
Tools for initialising parameters to emulate an IIR filter.
"""
import torch
from scipy import signal
from numpy.random import uniform
from hypercoil.functional.domain import Identity


class IIRFilterSpec(object):
    def __init__(self, Wn=None, N=1, ftype='butter', btype='bandpass', fs=None,
                 rp=0.1, rs=20, norm='phase'):
        self.Wn = Wn
        self.N = N
        self.ftype = ftype
        self.btype = btype
        self.fs = fs
        self.rp = rp
        self.rs = rs
        self.norm = norm

    def initialise_coefs(self, domain=None):
        if domain is not None:
            raise NotImplementedError('No domain support yet in IIRFilterSpec')
        domain = domain or Identity()
        if self.ftype == 'butter':
            iirfilter = signal.butter
            filter_params = {}
        elif self.ftype == 'cheby1':
            iirfilter = signal.cheby1
            filter_params = {'rp' : self.rp}
        elif self.ftype == 'cheby2':
            iirfilter = signal.cheby2
            filter_params = {'rs' : self.rs}
        elif self.ftype == 'ellip':
            iirfilter = signal.ellip
            filter_params = {'rp' : self.rp, 'rs' : self.rs}
        elif self.ftype == 'bessel':
            iirfilter = signal.bessel
            filter_params = {'norm' : self.norm}
        elif self.ftype == 'kuznetsov':
            self.coefs = kuznetsov_init(N=self.N, btype=self.btype)
            return
        else:
            raise ValueError(f'Unrecognised filter type : {self.ftype}')
        self.coefs = iirfilter_coefs(
            iirfilter=iirfilter,
            N=self.N,
            Wn = self.Wn,
            btype=self.btype,
            fs=self.fs,
            filter_params=filter_params
        )



def iirfilter_coefs(iirfilter, N, Wn, btype='bandpass', fs=None,
                    filter_params=None):
    import numpy as np
    filter_params = filter_params or {}
    N = _ensure_ndarray(N).astype(int)
    Wn = _ensure_ndarray(Wn)
    if btype in ('bandpass', 'bandstop') and Wn.ndim < 2:
        Wn = Wn.reshape(-1, 2)
    return [
        iirfilter(N=n, Wn=wn, btype=btype, fs=fs, **filter_params)
        for n, wn in zip(N, Wn)
    ]


def kuznetsov_init(N, btype='bandpass'):
    multiplier = 1
    if btype in ('bandpass', 'bandstop'):
        multiplier = 2
    #val, _ = torch.sort(torch.abs(X))
    b = uniform(-1, 1, N * multiplier + 1)
    a = uniform(-0.5, 0.5, N * multiplier + 1)
    a[0] = 1
    return b, a


def _ensure_ndarray(obj):
    """
    Ensure that the object is an iterable ndarray with dimension greater than
    or equal to 1. Another function we'd do well to get rid of in the future.
    """
    import numpy as np
    try:
        i = iter(obj)
        return np.array(obj)
    except TypeError:
        return np.array([obj])
