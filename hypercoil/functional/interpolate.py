# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interpolation
~~~~~~~~~~~~~
Methods for interpolating unseen or censored frames.
"""
import torch
from functools import partial
from .utils import conform_mask, apply_mask


def hybrid_interpolate(
    data,
    mask,
    max_weighted_stage=3,
    map_to_kernel=None,
    oversampling_frequency=8,
    maximum_frequency=1,
    frequency_thresh=0.3
):
    ##TODO
    # Right now, we're using the first weighted interpolation only for
    # determining the frames where spectral interpolation should be applied.
    # This seems rather wasteful.
    rec = weighted_interpolate(
        data=data,
        mask=mask,
        start_stage=1,
        max_stage=max_weighted_stage,
        map_to_kernel=None
    )
    spec_mask = ~torch.isnan(rec).sum(-2).to(torch.bool)
    rec = spectral_interpolate(
        data=data,
        tmask=spec_mask,
        oversampling_frequency=oversampling_frequency,
        maximum_frequency=maximum_frequency,
        sampling_period=1,
        thresh=frequency_thresh
    )
    final_mask = (mask + ~spec_mask).to(torch.bool).unsqueeze(-3)
    final_data = torch.where(final_mask, rec, data)
    rec = weighted_interpolate(
        data=final_data,
        mask=final_mask,
        start_stage=1,
        max_stage=max_weighted_stage + 1,
        map_to_kernel=None
    )
    return torch.where(final_mask, final_data, rec)


def weighted_interpolate(
    data,
    mask,
    start_stage=1,
    max_stage=None,
    map_to_kernel=None
):
    """
    Interpolate unseen time frames as a weighted average of neighbours.
    """
    batch_size = data.shape[0]
    if max_stage is None:
        max_stage = float('inf')
    if map_to_kernel is None:
        map_to_kernel = lambda stage: torch.ones(
            2 * stage + 1,
            dtype=data.dtype,
            device=data.device
        )
    cur_stage = 1
    rec = data
    mask = mask.view(batch_size, 1, 1, -1)
    rec_mask = mask
    while cur_stage < max_stage:
        kernel = map_to_kernel(cur_stage).view(1, 1, 1, -1)
        rec = reconstruct_weighted(
            rec,
            rec_mask.to(dtype=rec.dtype).view(batch_size, 1, 1, -1),
            kernel,
            cur_stage
        )
        rec_mask = ~torch.isnan(rec.sum((-2, -3), keepdim=True))
        rec = torch.where(mask, data, rec)
        rmask = conform_mask(rec, rec_mask, axis=-1, batch=True)
        rec[~rmask] = 0
        if (~rec_mask).sum() == 0:
            break
        cur_stage += 1
    rec[~rmask] = float('nan')
    return rec


def reconstruct_weighted(data, mask, kernel, stage):
    val = torch.conv2d(
        (data * mask),
        kernel,
        stride=1,
        padding=(0, stage)
    )
    wt = torch.conv2d(
        mask,
        kernel,
        stride=1,
        padding=(0, stage)
    )
    return val / wt


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
        msk = msk.squeeze()
        try:
            (sin_basis, cos_basis, angular_frequencies, all_samples
                ) = _periodogram_cfg(
                    tmask=msk,
                    sampling_period=sampling_period,
                    oversampling_frequency=oversampling_frequency,
                    maximum_frequency=maximum_frequency,
                    dtype=dtype,
                    device=device
                )
        except InterpolationError:
            continue
        recon[i] = _interpolate_spectral(
            data=apply_mask(tsr, msk, -1),
            sine_basis=sin_basis,
            cosine_basis=cos_basis,
            angular_frequencies=angular_frequencies,
            all_samples=all_samples,
            thresh=thresh
        )
    msk = conform_mask(data, tmask, axis=-1, batch=True)
    return torch.where(msk, data, recon)


class InterpolationError(Exception): pass


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
        raise InterpolationError(
            'No interpolation is necessary for this dataset.')

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
