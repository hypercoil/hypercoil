# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
IIR filter initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~
Tools for initialising parameters to match the frequency attenuation curve of
an IIR filter.
"""
import torch
import math


class IIRFilterSpec(object):
    def __init__(self, N, Wn, ftype='butter', btype='bandpass',
                 fs=None, rp=None, rs=None, norm='phase',
                 domain='atanh', bounds=(-3, 3)):
        self.N = N
        self.Wn = Wn
        self.ftype = ftype
        self.btype = btype
        self.fs = fs
        self.rp = rp
        self.rs = rs
        self.norm = norm
        self.domain = domain
        self.bounds = bounds
        self.n_filters = len(self.N)

    def initialise_spectrum(self, worN, ood='clip'):
        if self.ftype == 'butter':
            self.spectrum = butterworth_spectrum(
                N=self.N, Wn=self.Wn, btype=self.btype, worN=worN, fs=self.fs)
        if self.ftype == 'cheby1':
            self.spectrum = chebyshev1_spectrum(
                N=self.N, Wn=self.Wn, rp=self.rp, btype=self.btype,
                worN=worN, fs=self.fs)
        if self.ftype == 'cheby2':
            self.spectrum = chebyshev2_spectrum(
                N=self.N, Wn=self.Wn, rs=self.rs, btype=self.btype,
                worN=worN, fs=self.fs)
        if self.ftype == 'ellip':
            self.spectrum = elliptic_spectrum(
                N=self.N, Wn=self.Wn, rp=self.rp, rs=self.rs,
                worN=worN, btype=self.btype, fs=self.fs)
        if self.ftype == 'bessel':
            self.spectrum = bessel_spectrum(
                N=self.N, Wn=self.Wn, btype=self.btype,
                worN=worN, norm=self.norm, fs=self.fs)
        if self.ftype == 'ideal':
            self.spectrum = ideal_spectrum(
                N=self.N, Wn=self.Wn, btype=self.btype, worN=worN, fs=self.fs)
        self.transform_and_bound_spectrum(ood)

    def transform_and_bound_spectrum(self, ood):
        if self.domain == 'linear':
            return None

        ampl = torch.abs(spectrum)
        phase = torch.angle(spectrum)

        elif self.domain == 'atanh':
            ampl = self._handle_ood(ampl, bound=1, ood=ood)
            ampl = torch.atanh(ampl)
            self._bound_and_recompose()

    def _handle_ood(self, ampl, bound, ood):
        if ood == 'clip':
            ampl[ampl > bound] = bound
        elif ood == 'norm' and ampl.max(0) > bound:
            ampl /= (ampl.max(0) / bound)
        return ampl

    def _bound_and_recompose(self, ampl, phase):
        ampl[ampl < self.bounds[0]] = self.bounds[0]
        ampl[ampl > self.bounds[1]] = self.bounds[1]
        self.spectrum = ampl * torch.exp(phase * 1j)
