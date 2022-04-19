# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Fourier-domain filter
~~~~~~~~~~~~~~~~~~~~~
Convolve the signal via multiplication in the Fourier domain.
"""
import torch
import torch.fft
from functools import partial


def product_filter(X, weight, **params):
    r"""
    Convolve a multivariate signal via multiplication in the frequency domain.

    :Dimension: **Input :** :math:`(N, *, obs)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, obs denotes number of observations
                    per data channel.
                **Weight :** :math:`(*, \left\lfloor \frac{obs}{2} \right\rfloor + 1)`
                    As above.
                **Output :** :math:`(N, *, 2 \left\lfloor \frac{obs}{2} \right\rfloor )`
                    As above.

    Parameters
    ----------
    X : Tensor
        The (potentially multivariate) signal to be filtered. The final axis
        should correspond to the time domain of each signal or its analogue.
    weight : Tensor
        The filter gain at each frequency bin in the spectrum, ordered low to
        high along the last axis. Dimensions before the last can be used to
        apply different filters to different variables in the input signal
        according to tensor broadcasting rules.
    **params
        Any additional parameters provided will be passed to `torch.fft.rfft`
        and `torch.fft.irfft`.

    Returns
    -------
    out : Tensor
        Original time series filtered via multiplication in the frequency
        domain.
    """
    n = X.size(-1)
    Xf = torch.fft.rfft(X, n=n, **params)
    Xf_filt = weight * Xf
    return torch.fft.irfft(Xf_filt, n=n, **params)


def product_filtfilt(X, weight, **params):
    r"""
    Perform zero-phase digital filtering of a signal via multiplication in the
    frequency domain.

    This function operates by first filtering the data and then filtering a
    time-reversed copy of the filtered data again. Note that this doubles the
    effective filter order.

    If the `weight` argument is strictly real, then the filter has no phase
    delay component and it could make sense to simply use `product_filter`
    depending on the context. (The gradient could still have an imaginary
    component.)

    :Dimension: **Input :** :math:`(N, *, obs)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, obs denotes number of observations
                    per data channel.
                **Weight :** :math:`(*, \left\lfloor \frac{obs}{2} \right\rfloor + 1)`
                    As above.
                **Output :** :math:`(N, *, 2 \left\lfloor \frac{obs}{2} \right\rfloor )`
                    As above.

    Parameters
    ----------
    X : Tensor
        The (potentially multivariate) signal to be filtered. The final axis
        should correspond to the time domain of each signal or its analogue.
    weight : Tensor
        The filter gain at each frequency bin in the spectrum, ordered low to
        high along the last axis. Dimensions before the last can be used to
        apply different filters to different variables in the input signal
        according to tensor broadcasting rules.
    **params
        Any additional parameters provided will be passed to `torch.fft.rfft` and
        `torch.fft.irfft`.

    Returns
    -------
    out : Tensor
        Original time series filtered forward and backward via multiplication
        in the frequency domain.
    """
    X_filt = product_filter(X, weight, **params)
    return product_filter(X_filt.flip(-1), weight, **params).flip(-1)


def spectral_interpolate(
    data,
    tmask,
    oversampling_frequency=8,
    maximum_frequency=1
):
    dtype, device = data.dtype, data.device
    recon = torch.zeros_like(data, dtype=dtype, device=device)
    for i, (tsr, msk) in enumerate(zip(data, tmask)):
        (sin_term, cos_term, angular_frequencies, all_samples
            ) = _periodogram_cfg(
                tmask=msk,
                sampling_period=sampling_period,
                oversampling_frequency=oversampling_frequency,
                maximum_frequency=maximum_frequency,
                dtype=dtype,
                device=device
            )
        recon[i] = _interpolate_lombscargle(
            data=tsr,
            sine_term=sin_term,
            cosine_term=cos_term,
            angular_frequencies=angular_frequencies,
            all_samples=all_samples
        )
    return torch.where(tmask, data, recon)



def _periodogram_cfg(tmask,
                     sampling_period=1,
                     oversampling_frequency=8,
                     maximum_frequency=1,
                     dtype=None,
                     device=None):
    """
    Configure inputs for Lomb-Scargle interpolation.

    Parameters
    ----------
    tmask : tensor
        Boolean tensor indicating whether the value in each frame should
        be interpolated.
    sampling_period : float (default 1)
        The sampling period or repetition time.
    oversampling_frequency : int (default 8)
        Oversampling frequency for the periodogram.
    maximum_frequency : float
        The maximum frequency in the dataset, as a fraction of Nyquist.
        Default 1 (Nyquist).
    dtype : torch ``dtype`` (default None)
        Tensor data type.
    device : torch ``device`` (default None)
        Tensor device.

    Returns
    -------
    sin_term: numpy array
        Sine basis term for the periodogram.
    cos_term: numpy array
        Cosine basis term for the periodogram.
    angular_frequencies: numpy array
        Angular frequencies for computing the periodogram.
    all_samples: numpy array
        Temporal indices of all observations, seen and unseen.
    """
    n_samples = tmask.shape[-1]

    seen_samples = (
        torch.where(tmask)[0].to(dtype=dtype, device=device) + 1
    ) * sampling_period
    timespan = seen_samples.max() - seen_samples.min()
    freqstep = 1 / (timespan * oversampling_frequency)
    n_samples_seen = seen_samples.shape[0]
    if n_samples_seen == n_samples:
        raise ValueError('No interpolation is necessary for this dataset.')

    all_samples = torch.arange(start=sampling_period,
                               step=sampling_period,
                               end=sampling_period * (n_samples + 1),
                               dtype=dtype,
                               device=device)
    sampling_frequencies = torch.arange(
        start=freqstep,
        step=freqstep,
        end=(maximum_frequency * n_samples_seen / (2 * timespan) + freqstep),
        dtype=dtype,
        device=device
    )
    angular_frequencies = 2 * torch.pi * sampling_frequencies

    arg = angular_frequencies.view(-1, 1) @ seen_samples.view(1, -1)
    offsets = torch.atan2(
        torch.sin(2 * arg).sum(1),
        torch.cos(2 * arg).sum(1)
    ) / (2 * angular_frequencies)

    arg = arg - (angular_frequencies * offsets).view(-1, 1)
    cos_term = torch.cos(arg)
    sin_term = torch.sin(arg)
    return sin_term, cos_term, angular_frequencies, all_samples


def _compute_term(term, data):
    """
    Compute the transform from seen data for sin and cos terms.
    """
    mult = term.unsqueeze(-1) * data.transpose(-1, -2)
    # There seems to be a missing square here in the original implementation.
    # Putting it back, however, results in a much poorer fit.
    num = mult.sum(1)
    denom = (term ** 2).sum(1)
    return (num / denom.unsqueeze(-1))


def _reconstruct_term(
    term, fn, angular_frequencies, all_samples):
    """
    Interpolate over unseen epochs; reconstruct the time series.
    """
    term_prod = fn(angular_frequencies.view(-1, 1) @ all_samples.view(1, -1))
    term_recon = term_prod.unsqueeze(-1) @ term.unsqueeze(-2)
    return term_recon.sum(0)


def _interpolate_lombscargle(data,
                             sine_term,
                             cosine_term,
                             angular_frequencies,
                             all_samples):
    """
    Temporally interpolate over unseen (masked) values in a dataset using an
    approach based on the Lomb-Scargle periodogram. Follows code originally
    written in MATLAB by Anish Mitra and Jonathan Power:
    https://www.ncbi.nlm.nih.gov/pubmed/23994314
    The original code can be found in the function `getTransform` here:
        https://github.com/MidnightScanClub/MSC_Gratton2018_Codebase/blob/ ...
        master/FCProcessing/FCPROCESS_MSC_task.m

    Parameters
    ----------
    data: numpy array
        Seen data to use as a reference for reconstruction of unseen data.
    sine_term: numpy array
        Sine basis term for the periodogram.
    cosine_term: numpy array
        Cosine basis term for the periodogram.
    angular_frequencies: numpy array
        Angular frequencies for computing the periodogram.
    all_samples: numpy array
        Temporal indices of all samples, seen and unseen.

    Returns
    -------
    recon: numpy array
        Input data with unseen frames reconstructed via interpolation based on
        the Lomb-Scargle periodogram.
    """
    reconstruct = partial(
        _reconstruct_term,
        angular_frequencies=angular_frequencies,
        all_samples=all_samples
    )

    c = _compute_term(cosine_term, data)
    s = _compute_term(sine_term, data)

    s_recon = reconstruct(s, torch.sin)
    c_recon = reconstruct(c, torch.cos)
    recon = (c_recon + s_recon).transpose(-1, -2)

    # Normalise the reconstructed spectrum. This is necessary when the
    # oversampling frequency exceeds 1.
    std_recon = recon.std(-1, keepdim=True, unbiased=False)
    std_orig = data.std(-1, keepdim=True, unbiased=False)
    norm_fac = std_recon / std_orig

    return recon / norm_fac
