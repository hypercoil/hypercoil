# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Methods for interpolating, extrapolating, and imputing unseen or censored
frames.
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
    frequency_thresh=0.3,
    handle_fail='orig'
):
    """
    Interpolate unseen time frames using a hybrid approach that combines
    :func:`weighted <weighted_interpolate>` and
    :func:`spectral <spectral_interpolate>`
    methods.

    Hybrid interpolation uses the :func:`weighted <weighted_interpolate>`
    method for unseen frames that are no more than ``max_weighted_stage``
    frames away from a seen frame, and the
    :func:`spectral <spectral_interpolate>` method otherwise.
    (More specifically, if ``map_to_kernel`` is set, it uses the weighted
    approach if the weighted approach successfully imputes using the kernel
    returned when the argument to ``map_to_kernel`` is
    ``max_weighted_stage``.) Imputation proceeds as follows:

    * Unseen frames are divided into two groups according to the approach that
      will be used for imputation.
    * Spectral interpolation is applied using the seen time frames.
    * Weighted interpolation is applied using the seen time frames, together
      with the frames interpolated using the spectral method.

    Parameters
    ----------
    data : tensor
        Time series data.
    mask : boolean tensor
        Boolean tensor indicating whether the value in each frame of the input
        time series is observed. ``True`` indicates that the original data are
        "good" or observed, while ``False`` indicates that they are "bad" or
        missing and flags them for interpolation.
    max_weighted_stage : int or None (default 3)
        The final stage of weighted interpolation. The meaning of this is
        governed by the ``map_to_kernel`` argument. If no ``map_to_kernel``
        argument is otherwise specified, it sets the maximum size of a boxcar
        window for averaging. By default, a maximum stage of 3 is specified
        for weighted interpolation; any unseen frames that cannot be imputed
        using this maximum are instead imputed using the spectral approach.
    map_to_kernel : callable(int -> tensor)
        A function that uses the integer value of the current stage to create
        a convolutional kernel for weighting of neighbours. By default, a
        boxcar window that includes the current frame, together with ``stage``
        frames in each of the forward and backward directions, is returned.
    oversampling_frequency : float (default 8)
        Determines the number of frequency bins to use when estimating the
        sine and cosine spectra in spectral interpolation. 1 indicates that
        the number of bins should be the same as in a Fourier transform,
        while larger values indicate that frequency bins should be
        oversampled.
    maximum_frequency : float (default 1)
        Maximum frequency bin to consider in the spectral fit, as a fraction
        of Nyquist.
    frequency_thresh : float (default 0.3)
        Because of the non-orthogonality of the basis functions, spurious
        variance will often be captured in the spectral estimates. To control
        this spurious variance, all frequency bins whose estimates are less
        than ``thresh``, as a fraction of the maximum estimate across all
        bins, are set to 0.
    handle_fail : ``'raise'`` or ``'orig'`` (default ``'orig'``)
        Specifies behaviour if the interpolation fails (which typically occurs
        if every frame is labelled as unseen). ``'raise'`` raises an
        exception, while ``'orig'`` returns values from the input time series
        for any frames that are still missing after interpolation.

    Returns
    -------
    tensor
        Input dataset with missing frames imputed.
    """
    batch_size = data.shape[0]
    ##TODO
    # Right now, we're using the first weighted interpolation only for
    # determining the frames where spectral interpolation should be applied.
    # This seems rather wasteful.
    #print(torch.where(torch.isnan(data)))
    rec = weighted_interpolate(
        data=data,
        mask=mask,
        start_stage=1,
        max_stage=max_weighted_stage,
        map_to_kernel=None
    )
    #print(torch.where(torch.isnan(rec)))
    spec_mask = ~torch.isnan(rec).sum(-2).to(torch.bool)
    rec = spectral_interpolate(
        data=data,
        tmask=spec_mask,
        oversampling_frequency=oversampling_frequency,
        maximum_frequency=maximum_frequency,
        sampling_period=1,
        thresh=frequency_thresh,
        handle_fail=handle_fail
    )
    mask = mask.squeeze()
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    final_mask = (
        mask.view(batch_size, -1) +
        ~spec_mask.view(batch_size, -1)).to(torch.bool)
    final_mask = (
        final_mask * ~torch.isnan(rec).squeeze().sum(-2, keepdim=True)
    ).to(torch.bool).squeeze()
    final_mask = conform_mask(data, final_mask, axis=-1, batch=True)
    final_data = torch.where(
        final_mask,
        rec,
        data)
    rec = weighted_interpolate(
        data=final_data,
        mask=final_mask.squeeze(),
        start_stage=1,
        max_stage=None,
        map_to_kernel=None
    )
    if handle_fail == 'orig':
        rec = torch.where(final_mask, final_data, rec)
        rec = torch.where(torch.isnan(rec), data, rec)
        if torch.any(torch.isnan(rec)):
            raise InterpolationError(
                'Data are still missing after interpolation. This typically '
                'occurs when all input data are NaN-valued.')
        return rec
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

    Interpolation proceeds iteratively over progressively longer window sizes.
    It first defines a convolutional weighting/window kernel for the current
    window size, and then sets the values of unseen time frames to the
    convolution of seen time frames with this kernel, and marks those time
    frames as seen for the next iteration. Iteration proceeds either until the
    specified maximum stage or until every unseen frame is imputed.

    Parameters
    ----------
    data : tensor
        Time series data.
    mask : boolean tensor
        Boolean tensor indicating whether the value in each frame of the input
        time series is observed. ``True`` indicates that the original data are
        "good" or observed, while ``False`` indicates that they are "bad" or
        missing and flags them for interpolation.
    start_stage : int
        The first stage of weighted interpolation. The meaning of this is
        governed by the ``map_to_kernel`` argument. If no ``map_to_kernel``
        argument is otherwise specified, it sets the initial size of a boxcar
        window for averaging. (A value of 1 corresponds to averaging over the
        current frame, together with 1 frame in each of the forward and
        reverse directions.)
    max_stage : int or None (default None)
        The final stage of weighted interpolation. The meaning of this is
        governed by the ``map_to_kernel`` argument. If no ``map_to_kernel``
        argument is otherwise specified, it sets the maximum size of a boxcar
        window for averaging. By default, no maximum size is specified, and
        iteration proceeds until every unseen time point is imputed.
    map_to_kernel : callable(int -> tensor)
        A function that uses the integer value of the current stage to create
        a convolutional kernel for weighting of neighbours. By default, a
        boxcar window that includes the current frame, together with ``stage``
        frames in each of the forward and backward directions, is returned.

    Returns
    -------
    tensor
        Input dataset with missing frames imputed using the specified weighted
        average.
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
    orig_mask = conform_mask(rec, mask, axis=-1, batch=True)
    rec_mask = conform_mask(rec, mask, axis=-1, batch=True)
    while cur_stage < max_stage:
        kernel = map_to_kernel(cur_stage).view(1, 1, 1, -1)
        #rec_old = rec
        rec = reconstruct_weighted(
            rec,
            rec_mask,
            kernel,
            cur_stage
        )
        #rec, rec_mask = _get_data_for_weighted_recon(rec_old, rec, rec_mask)
        rec, rec_mask = _get_data_for_weighted_recon(data, rec, orig_mask)
        if (~rec_mask).sum() == 0:
            break
        cur_stage += 1
    rec[~rec_mask] = float('nan')
    return rec


def _get_data_for_weighted_recon(orig_data, rec_data, mask):
    data = torch.where(mask, orig_data, rec_data)
    mask = ~torch.isnan(rec_data.sum((-2, -3)))
    mask = conform_mask(data, mask, axis=-1, batch=True)
    data[~mask] = 0
    return data, mask


def reconstruct_weighted(data, mask, kernel, stage):
    padding = kernel.shape[-1] // 2
    mask = mask.to(dtype=data.dtype, device=data.device)
    val = torch.conv2d(
        (data * mask),
        kernel,
        stride=1,
        padding=(0, padding)
    )
    wt = torch.conv2d(
        mask,
        kernel,
        stride=1,
        padding=(0, padding)
    )
    if torch.all(torch.isnan(data)): assert 0
    return val / wt


def spectral_interpolate(
    data,
    tmask,
    oversampling_frequency=8,
    maximum_frequency=1,
    sampling_period=1,
    thresh=0,
    handle_fail='raise'
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
        except RuntimeError:
            ##TODO: this is a critical unit test.
            # We need a principled way of handling different data cases, such
            # all missing (which can and will occur under windowing conditions)
            if handle_fail == 'orig':
                #print('Disaster!')
                #print(recon[i].shape, tsr.shape)
                recon[i] = tsr
                continue
            else:
                raise RuntimeError('The dataset provided likely has no seen time points')
        recon[i] = _interpolate_spectral(
            data=apply_mask(tsr, msk, -1),
            sine_basis=sin_basis,
            cosine_basis=cos_basis,
            angular_frequencies=angular_frequencies,
            all_samples=all_samples,
            thresh=thresh
        )
    msk = conform_mask(data, tmask.squeeze(), axis=-1, batch=True)
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
    norm_fac = std_recon / (std_orig + torch.finfo(data.dtype).eps)

    return recon / (norm_fac + torch.finfo(data.dtype).eps)
