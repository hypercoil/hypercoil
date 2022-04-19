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
from .utils import mask as apply_mask


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
    maximum_frequency=1,
    sampling_period=1,
    thresh=0
):
    """
    Spectral interpolation based on basis function projection.

    Given a time series with missing or corrupt observations, this operation
    estimates the time series frequency spectra by proxy of projections onto
    sine and cosine basis functions, and then uses the frequency spectrum
    estimates as basis function weights in the reconstruction of missing
    observations.

    .. warning::
        Users of this approach must be advised that the missing observations
        result in non-orthogonal basis functions, and accordingly the variance
        captured by different frequency bins might be shared.

    .. note::
        This method is inspired by
        `previous work from Anish Mitra and Jonathan Power
        <https://github.com/MidnightScanClub/MSCcodebase/blob/master/Processing/FCPROCESS_MSC.m>`_
        which in turn is inspired by the Lomb-Scargle periodogram.

    This method is used to temporarily interpolate over time points flagged
    for omission due to high artefact content, so as to reduce disruptions to
    the autocorrelation structure introduced by either artefactual outliers or
    zeroing (or arguably linearly interpolating) those outliers. This is
    useful for operations such as filtering and convolution for which the
    autocorrelation structure of the data is relevant. It should not be used
    for operations in which the autocorrelation structure plays no role, such
    as covariance estimation, for which the weights of missing observations
    can easily be set to 0.

    .. warning::
        ``spectral_interpolate`` expects batched input ``data`` and ``tmask``
        arguments. If your inputs are not batched, ``unsqueeze`` their first
        dimension before providing them as inputs.

    Parameters
    ----------
    data : tensor
        Time series data.
    tmask : boolean tensor
        Boolean tensor indicating whether the value in each frame of the input
        time series is observed. ``True`` indicates that the original data are
        "good" or observed, while ``False`` indicates that they are "bad" or
        missing and flags them for interpolation.
    oversampling_frequency : float (default 8)
        Determines the number of frequency bins to use when estimating the
        sine and cosine spectra. 1 indicates that the number of bins should be
        the same as in a Fourier transform, while larger values indicate that
        frequency bins should be oversampled.
    maximum_frequency : float (default 1)
        Maximum frequency bin to consider in the fit, as a fraction of
        Nyquist.
    sampling_period : float (default 1)
        Period separating consecutive samples in ``data``.
    thresh : float (default 0)
        Because of the non-orthogonality of the basis functions, spurious
        variance will often be captured in the spectral estimates. To control
        this spurious variance, all frequency bins whose estimates are less
        than ``thresh``, as a fraction of the maximum estimate across all
        bins, are set to 0.

    Returns
    -------
    tensor
        Input ``data`` whose flagged frames are replaced using the spectral
        interpolation procedure.
    """
    dtype, device = data.dtype, data.device
    recon = torch.zeros_like(data, dtype=dtype, device=device)
    for i, (tsr, msk) in enumerate(zip(data, tmask)):
        (sin_basis, cos_basis, angular_frequencies, all_samples
            ) = _periodogram_cfg(
                tmask=msk,
                sampling_period=sampling_period,
                oversampling_frequency=oversampling_frequency,
                maximum_frequency=maximum_frequency,
                dtype=dtype,
                device=device
            )
        recon[i] = _interpolate_spectral(
            data=apply_mask(tsr, msk, -1),
            sine_basis=sin_basis,
            cosine_basis=cos_basis,
            angular_frequencies=angular_frequencies,
            all_samples=all_samples,
            thresh=thresh
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
    sin_basis: numpy array
        Sine basis term for the periodogram.
    cos_basis: numpy array
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
    basis_frequencies = torch.arange(
        start=freqstep,
        step=freqstep,
        end=(maximum_frequency * n_samples_seen / (2 * timespan) + freqstep),
        dtype=dtype,
        device=device
    )
    angular_frequencies = 2 * torch.pi * basis_frequencies

    arg = angular_frequencies.view(-1, 1) @ seen_samples.view(1, -1)
    ##TODO: adding the offset term worsens the performance: Why?
    #offsets = torch.atan2(
    #    torch.sin(2 * arg).sum(1),
    #    torch.cos(2 * arg).sum(1)
    #) / (2 * angular_frequencies)
    #arg = arg - (angular_frequencies * offsets).view(-1, 1)

    cos_basis = torch.cos(arg)
    sin_basis = torch.sin(arg)
    return sin_basis, cos_basis, angular_frequencies, all_samples


def _fit_spectrum(basis, data, thresh=0):
    """
    Compute the transform from seen data for sin and cos terms.
    Here we project the data onto each of the sine and cosine bases.
    Note that, due to missing observations, the basis functions are not
    exactly orthogonal. Thus, we will have some shared variance captured by
    our estimates.
    """
    num = basis @ data.transpose(-1, -2)
    # There seems to be a missing square here in the original implementation.
    # Putting it back, however, results in a much poorer fit. Here we're
    # instead going with projecting the seen time series onto each of the sin
    # and cos terms to get our spectra.
    denom = torch.sqrt((basis ** 2).sum(-1))
    spectrum = (num / denom.unsqueeze(-1))
    spectrum[(spectrum.abs() / spectrum.abs().max()) <= thresh] = 0
    return spectrum


def _reconstruct_from_spectrum(
    spectrum, fn, angular_frequencies, all_samples):
    """
    Interpolate over unseen epochs; reconstruct the time series.
    """
    basis = fn(angular_frequencies.view(-1, 1) @ all_samples.view(1, -1))
    return basis.transpose(-1, -2) @ spectrum


def _interpolate_spectral(data,
                          sine_basis,
                          cosine_basis,
                          angular_frequencies,
                          all_samples,
                          thresh=0):
    """
    Temporally interpolate over unseen (masked) values in a dataset using an
    approach loosely inspired by the Lomb-Scargle periodogram. Modified from
    code originally written in MATLAB by Anish Mitra and Jonathan Power that
    likely followed the original periodogram more closely:
    https://www.ncbi.nlm.nih.gov/pubmed/23994314
    The original code can be found in the function `getTransform` here:
        https://github.com/MidnightScanClub/MSC_Gratton2018_Codebase/blob/ ...
        master/FCProcessing/FCPROCESS_MSC_task.m

    In summary, the approach involves first projecting seen data onto sine and
    cosine bases evaluated at the seen time points in order to estimate fit
    coefficients, and then applying the estimated coefficients to the full
    sine and cosine basis functions over all time points.

    Parameters
    ----------
    data : numpy array
        Seen data to use as a reference for reconstruction of unseen data.
    sine_basis : numpy array
        Sine basis term for the periodogram.
    cosine_basis : numpy array
        Cosine basis term for the periodogram.
    angular_frequencies: numpy array
        Angular frequencies for computing the periodogram.
    all_samples : numpy array
        Temporal indices of all samples, seen and unseen.
    thresh : float
        Threshold applied to sine and cosine spectral terms to remove weak or
        spurious bands.

    Returns
    -------
    recon: numpy array
        Input data reconstructed based on spectral projection estimates.
    """
    reconstruct = partial(
        _reconstruct_from_spectrum,
        angular_frequencies=angular_frequencies,
        all_samples=all_samples
    )

    c = _fit_spectrum(cosine_basis, data, thresh=thresh)
    s = _fit_spectrum(sine_basis, data, thresh=thresh)

    s_recon = reconstruct(s, torch.sin)
    c_recon = reconstruct(c, torch.cos)
    recon = (c_recon + s_recon).transpose(-1, -2)

    # Normalise the reconstructed spectrum. This is necessary when the
    # oversampling frequency exceeds 1.
    std_recon = recon.std(-1, keepdim=True, unbiased=False)
    std_orig = data.std(-1, keepdim=True, unbiased=False)
    norm_fac = std_recon / std_orig

    return recon / norm_fac
