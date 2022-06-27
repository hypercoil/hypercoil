# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Convolve the signal via multiplication in the Fourier domain.
"""
import torch
import torch.fft
from .cov import corr
from .utils import complex_decompose, orient_and_conform


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
        Any additional parameters provided will be passed to ``torch.fft.rfft``
        and ``torch.fft.irfft``.

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
    time-reversed copy of the filtered data again. Note that the effect on the
    amplitude is quadratic in the filter weight.

    If the ``weight`` argument is strictly real, then the filter has no phase
    delay component and it could make sense to simply use ``product_filter``
    depending on the context.

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
    out = product_filter(X_filt.flip(-1), weight, **params).flip(-1)
    return out


def analytic_signal(X, axis=-1, n=None):
    if X.is_complex():
        raise ValueError(
            'Input for analytic signal must be strictly real')

    Xf = torch.fft.fft(X, n=n, axis=axis)
    n = n or X.size(axis)
    h = torch.zeros(n, dtype=X.dtype, device=X.device)

    if n % 2 == 0:
        h[0] = h[n // 2] = 1
        h[1:(n // 2)] = 2
    else:
        h[0] = 1
        h[1:((n + 1) // 2)] = 2

    if Xf.dim() >= 1:
        h = orient_and_conform(h, axis=axis, reference=Xf)
        print(h.shape, Xf.shape)
    return torch.fft.ifft(Xf * h, axis=axis)


def ampl_phase_corr(
    X, weight=None, corr_axes=(0,), cov=corr, **params):
    """
    Covariance among frequency bins, amplitude and phase. To run it across the
    batch and region axes, use
    ``ampl_phase_corr(X, weight=None, corr_axes=(0, -2))``
    Be advised: there is no interesting structure here.
    """
    n = X.size(-1)
    Xf = torch.fft.rfft(X, n=n, **params)
    ampl, phase = complex_decompose(Xf)
    axes = [True for _ in ampl.shape]
    for ax in corr_axes:
        axes[ax] = False
    shape = [e for i, e in enumerate(ampl.shape) if axes[i]]
    new_axes = [i - len(corr_axes) for i in range(len(corr_axes))]
    ampl = torch.moveaxis(ampl, corr_axes, new_axes).reshape(*shape, -1)
    phase = torch.moveaxis(phase, corr_axes, new_axes).reshape(*shape, -1)
    return cov(ampl, weight=weight), cov(phase, weight=weight)
