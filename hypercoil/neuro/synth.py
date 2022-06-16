# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Synthesise data matching spectral and covariance properties of a reference.
"""
import torch
from ..functional import (
    complex_decompose
)
from ..data.functional import normalise


def match_spectra(signal, reference, use_mean=False, frequencies=False):
    if not frequencies:
        signal = torch.fft.rfft(signal)
        reference = torch.fft.rfft(reference)
    ampl_ref, _ = complex_decompose(reference)
    if use_mean:
        ampl_ref = ampl_ref.mean(0)
    matched = signal * ampl_ref
    ampl_matched, _ = complex_decompose(matched)
    matched = matched * ampl_ref.mean() / ampl_matched.mean()
    return torch.fft.irfft(matched)


def match_covariance(signal, reference, cov=False):
    if not cov:
        reference = reference.cov()
    L, Q = torch.linalg.eigh(reference)
    L_sqrt = L.sqrt()
    return Q @ (L_sqrt.unsqueeze(-1) * signal)


def match_reference(signal, reference, use_mean=False):
    matched = match_spectra(signal, reference, use_mean=use_mean)
    matched = normalise(matched)
    return match_covariance(signal=matched, reference=reference)


def match_cov_and_spectrum(signal, spectrum, cov):
    matched = match_spectra(
        signal=torch.fft.rfft(signal),
        reference=spectrum,
        frequencies=True
    )
    matched = normalise(matched)
    return match_covariance(
        signal=matched,
        reference=cov,
        cov=True
    )


def synthesise_matched(reference, use_mean=False):
    synth = torch.randn_like(reference)
    return match_reference(
        signal=synth,
        reference=reference,
        use_mean=use_mean
    )


def synthesise_from_cov_and_spectrum(spectrum, cov, dtype=None, device=None):
    n_ts = cov.size(-1)
    n_obs = 2 * (spectrum.size(-1) - 1)
    synth = torch.randn((n_ts, n_obs), dtype=dtype, device=device)
    return match_cov_and_spectrum(
        signal=synth,
        spectrum=spectrum,
        cov=cov
    )
