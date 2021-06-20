# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Fourier-domain filter
~~~~~~~~~~~~~~~~~~~~~
Convolve the signal via multiplication in the Fourier domain.
"""
import torch.fft


def product_filter(X, weight, **params):
    """
    Convolve a multivariate signal via multiplication in the frequency domain.

    Dimension
    ---------
    - Input: :math:`(N, *, obs)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      obs denotes number of observations
    - Weight: :math:`(*, \left\lfloor \frac{obs}{2} \right\rfloor + 1)`
    - Output: :math:`(N, *, 2 \left\lfloor \frac{obs}{2} \right\rfloor )`

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
    Any additional parameters provided will be passed to `torch.fft.rfft` and
        `torch.fft.irfft`.

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
    """
    Perform zero-phase digital filtering of a signal via multiplication in the
    frequency domain.

    This function operates by first filtering the data and then filtering a
    time-reversed copy of the filtered data again. Note that this doubles the
    effective filter order.

    If the `weight` argument is strictly real, then the filter has no phase
    delay component and it could make sense to simply use `product_filter`
    depending on the context. (The gradient could still have an imaginary
    component.)

    Dimension
    ---------
    - Input: :math:`(N, *, obs)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      obs denotes number of observations
    - Weight: :math:`(*, \left\lfloor \frac{obs}{2} \right\rfloor + 1)`
    - Output: :math:`(N, *, 2 \left\lfloor \frac{obs}{2} \right\rfloor )`

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