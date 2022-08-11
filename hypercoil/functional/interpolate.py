# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Methods for interpolating, extrapolating, and imputing unseen or censored
frames.
"""
import torch
import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Literal, Optional, Sequence, Tuple, Union
from .utils import conform_mask, vmap_over_outer, Tensor
from .tsconv import atleast_4d, conv, tsconv2d


class InterpolationError(Exception): pass


def hybrid_interpolate(
    data: Tensor,
    mask: Tensor,
    max_weighted_stage: int = 3,
    map_to_kernel: Optional[Callable] = None,
    oversampling_frequency: float = 8,
    maximum_frequency: float = 1,
    frequency_thresh: float = 0.3,
    handle_fail: Literal['orig', 'raise'] = 'orig'
) -> Tensor:
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
        map_to_kernel=map_to_kernel
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
        map_to_kernel=map_to_kernel
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


def _number_consecutive_impl(carry: int, x: Tensor) -> Tuple[int, int]:
    carry = jax.lax.cond(x, lambda c: c + 1, lambda c: 0, carry)
    return carry, carry


def number_consecutive(x: Tensor) -> Tensor:
    return jax.lax.scan(_number_consecutive_impl, 0, x)[1]


def max_number_consecutive(x: Tensor) -> Tensor:
    return number_consecutive(x).max()


def weighted_interpolate(
    data: Tensor,
    mask: Tensor,
    start_stage: int = 1,
    max_stage: Union[int, Literal['auto']] = 'auto',
    stages: Optional[Sequence[int]] = None,
    map_to_kernel: Optional[Callable] = None
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
    data = atleast_4d(data)
    mask = atleast_4d(mask)
    orig_data = data.copy()
    if stages is None:
        stages = all_stages(start_stage, max_stage, mask)
    max_stage = stages[-1]
    if map_to_kernel is None:
        map_to_kernel = partial(centred_square_kernel, max_stage=max_stage)
        #map_to_kernel = lambda s: jnp.ones(2 * s + 1)
    kernels = jnp.stack(make_kernels(stages, map_to_kernel))
    f = lambda x, k: _weighted_interpolate_stage(
        data=x[0],
        mask=x[1],
        kernel=k,
        #orig_data=orig_data,
    )
    (data, mask), _ = jax.lax.scan(
        f=f,
        init=(data, mask),
        xs=kernels,
    )
    #data = jnp.where(mask, data, float('nan'))
    return data


def make_kernels(
    stages: Sequence[int],
    map_to_kernel: Callable
) -> Sequence[Tensor]:
    return [map_to_kernel(stage) for stage in stages]


def all_stages(
    start_stage: int = 1,
    max_stage: Union[int, Literal['auto']] = 'auto',
    mask: Optional[Tensor] = None
) -> Sequence[int]:
    if max_stage == 'auto':
        max_stage = vmap_over_outer(max_number_consecutive, 1)((~mask,)).max()
    return range(start_stage, max_stage + 1)


def _weighted_interpolate_stage(
    data: Tensor,
    mask: Tensor,
    kernel: Tensor,
    #orig_data: Tensor,
):
    print(data)
    print(mask)
    print(kernel)
    stage_rec_data = reconstruct_weighted(
        data,
        mask,
        kernel
    )
    return _get_data_for_weighted_recon(data, stage_rec_data, mask), None


def centred_square_kernel(stage: int, max_stage: int) -> Tensor:
    kernel = jnp.zeros(2 * max_stage + 1)
    midpt = max_stage
    return kernel.at[(midpt - stage):(midpt + stage + 1)].set(1)


def _get_data_for_weighted_recon(
    orig_data: Tensor,
    rec_data: Tensor,
    mask: Tensor
) -> Tuple[Tensor, Tensor]:
    data = jnp.where(mask, orig_data, rec_data)
    mask = ~jnp.isnan(rec_data.sum((-2, -3), keepdims=True))
    data = jnp.where(mask, data, 0)
    return data, mask


def reconstruct_weighted(
    data: Tensor,
    mask: Tensor,
    kernel: Tensor
) -> Tensor:
    padding = kernel.shape[-1] // 2
    val = tsconv2d(jnp.where(mask, data, 0), kernel)
    wt = tsconv2d(jnp.where(mask, 1., 0.), kernel)
    return val / wt # (wt + jnp.finfo(wt.dtype).eps)


def spectral_interpolate(
    data: Tensor,
    tmask: Tensor,
    oversampling_frequency: float = 8,
    maximum_frequency: float = 1,
    sampling_period: float = 1,
    thresh: float = 0
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

    .. warning::
        If the input time series contains either no or all missing
        observations, then the output time series will be identical to the
        input time series.

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
    data = atleast_4d(data)
    tmask = atleast_4d(tmask)
    angular_frequencies, all_samples = _periodogram_cfg(
        n_samples=data.shape[-1],
        sampling_period=sampling_period,
        oversampling_frequency=oversampling_frequency,
        maximum_frequency=maximum_frequency
    )
    recon = vmap_over_outer(partial(
        _spectral_interpolate_single,
        all_samples=all_samples,
        angular_frequencies=angular_frequencies,
        thresh=thresh,
    ), 3)((data, tmask))
    msk = conform_mask(data, tmask.squeeze(), axis=-1, batch=True)
    return jnp.where(msk, data, recon)


def _spectral_interpolate_single(
    tsr: Tensor,
    msk: Tensor,
    all_samples: Tensor,
    angular_frequencies: Tensor,
    thresh: float = 0,
) -> Tensor:
    msk = msk.squeeze()
    def fn() -> Tensor:
        return _spectral_interpolate_single_impl(
            tsr,
            msk,
            all_samples,
            angular_frequencies,
            thresh
        )
    degenerate_mask = jnp.logical_or(msk.all(), (~msk).all())
    return jax.lax.cond(degenerate_mask, lambda: tsr, fn)


def _spectral_interpolate_single_impl(
    tsr: Tensor,
    msk: Tensor,
    all_samples: Tensor,
    angular_frequencies: Tensor,
    thresh: float = 0
) -> Tensor:
    sin_basis, cos_basis = _apply_periodogram(
        tmask=msk,
        all_samples=all_samples,
        angular_frequencies=angular_frequencies
    )
    return _interpolate_spectral(
        data=tsr,
        tmask=msk,
        sine_basis=sin_basis,
        cosine_basis=cos_basis,
        angular_frequencies=angular_frequencies,
        all_samples=all_samples,
        thresh=thresh
    )


def _apply_periodogram(
    tmask: Tensor,
    all_samples: Tensor,
    angular_frequencies: Tensor,
) -> Tuple[Tensor, Tensor]:
    seen_samples = jnp.where(tmask, all_samples, float('nan'))
    arg = jnp.outer(angular_frequencies, seen_samples)
    return jnp.sin(arg), jnp.cos(arg)


def _periodogram_cfg(
    n_samples: int,
    sampling_period: float = 1,
    oversampling_frequency: float = 8,
    maximum_frequency: float = 1,
) -> Tuple[Tensor, Tensor]:
    timespan = sampling_period * (n_samples + 1) - 1
    all_samples = jnp.arange(start=sampling_period,
                             step=sampling_period,
                             stop=timespan + 1)
    freqstep = 1 / (timespan * oversampling_frequency)

    angular_frequencies = 2 * jnp.pi * jnp.arange(
        start=freqstep,
        step=freqstep,
        stop=(maximum_frequency * n_samples / (2 * timespan) + freqstep)
    )
    return angular_frequencies, all_samples


def _fit_spectrum(
    basis: Tensor,
    data: Tensor,
    thresh: float = 0
) -> Tensor:
    """
    Compute the transform from seen data for sin and cos terms.
    Here we project the data onto each of the sine and cosine bases.
    Note that, due to missing observations, the basis functions are not
    exactly orthogonal. Thus, we will have some shared variance captured by
    our estimates. We can potentially mitigate this using a threshold or
    lateral inhibition.
    """
    num = basis @ data.swapaxes(-1, -2)
    # There seems to be a missing square here in the original implementation.
    # Putting it back, however, results in a much poorer fit. Here we're
    # instead going with projecting the seen time series onto each of the sin
    # and cos terms to get our spectra.
    denom = jnp.sqrt((basis ** 2).sum(-1, keepdims=True))
    spectrum = (num / denom)
    if thresh > 0:
        absval = jnp.abs(spectrum)
        mask = ((absval / absval.max()) <= thresh)
        return jnp.where(mask, 0, spectrum)
    return spectrum


def _reconstruct_from_spectrum(
    spectrum: Tensor,
    fn: Callable,
    angular_frequencies: Tensor,
    all_samples: Tensor
) -> Tensor:
    """
    Interpolate over unseen epochs; reconstruct the time series.
    """
    basis = fn(jnp.outer(angular_frequencies, all_samples))
    return basis.swapaxes(-1, -2) @ spectrum


def _interpolate_spectral(
    data: Tensor,
    tmask: Tensor,
    sine_basis: Tensor,
    cosine_basis: Tensor,
    angular_frequencies: Tensor,
    all_samples: Tensor,
    thresh: float = 0
) -> Tensor:
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

    data = jnp.where(tmask, data, 0)
    sine_basis = jnp.where(tmask, sine_basis, 0)
    cosine_basis = jnp.where(tmask, cosine_basis, 0)

    c = _fit_spectrum(cosine_basis, data, thresh=thresh)
    s = _fit_spectrum(sine_basis, data, thresh=thresh)

    s_recon = reconstruct(s, jnp.sin)
    c_recon = reconstruct(c, jnp.cos)
    recon = (c_recon + s_recon).swapaxes(-1, -2)

    # Normalise the reconstructed spectrum by projecting the seen time points
    # onto the real data. This will give us a beta value that we can use to
    # normalise the reconstructed time series.
    recon_seen = jnp.where(tmask, recon, 0)
    seen_data = jnp.where(tmask, data, 0)
    norm_fac = jnp.sum(recon_seen * seen_data, axis=-1) / \
        jnp.sum(seen_data ** 2, axis=-1)

    return recon / (norm_fac + jnp.finfo(data.dtype).eps)
