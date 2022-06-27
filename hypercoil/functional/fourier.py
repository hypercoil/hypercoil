# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Convolve the signal via multiplication in the Fourier domain.
"""
import math
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
        Any additional parameters provided will be passed to `torch.fft.rfft`
        and `torch.fft.irfft`.

    Returns
    -------
    out : Tensor
        Original time series filtered forward and backward via multiplication
        in the frequency domain.
    """
    X_filt = product_filter(X, weight, **params)
    out = product_filter(X_filt.flip(-1), weight, **params).flip(-1)
    return out


def unwrap(phase, axis=-1, discont=None, period=(2 * math.pi)):
    dd = phase.diff(axis=axis)
    half_period = period / 2
    if discont is None:
        discont = half_period

    slice1 = [slice(None, None)] * phase.dim()
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)

    interval_high = half_period
    interval_low = -interval_high
    ddmod = (dd - interval_low) % period + interval_low
    ddmod = torch.where(
        (ddmod == interval_low) & (dd > 0),
        torch.tensor(interval_high, dtype=phase.dtype, device=phase.device),
        ddmod
    )
    phase_correct = ddmod - dd
    phase_correct = torch.where(
        dd.abs() < discont,
        torch.tensor(0, dtype=phase.dtype, device=phase.device),
        phase_correct
    )
    unwrapped_phase = phase.clone()
    unwrapped_phase[slice1] = phase[slice1] + phase_correct.cumsum(axis)
    return unwrapped_phase


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
    return torch.fft.ifft(Xf * h, axis=axis)


def hilbert_transform(X, axis=-1, n=None):
    return analytic_signal(X=X, axis=axis, n=n).imag


def envelope(X, axis=-1, n=None):
    return analytic_signal(X=X, axis=axis, n=n).abs()


def instantaneous_phase(X, axis=-1, n=None):
    return unwrap(analytic_signal(X=X, axis=axis, n=n).angle())


def instantaneous_frequency(X, axis=-1, n=None, fs=1, period=(2 * math.pi)):
    inst_phase = instantaneous_phase(X=X, axis=axis, n=n).diff(dim=axis)
    return fs * inst_phase / period


def env_inst_freq(X, axis=-1, n=None, fs=1,
                  period=(2 * math.pi),
                  return_instantaneous_phase=False):
    Xa = analytic_signal(X=X, axis=axis, n=n)
    env = Xa.abs()
    inst_phase = unwrap(Xa.angle())
    inst_freq = fs * inst_phase.diff(dim=axis) / period
    if return_instantaneous_phase:
        return env, inst_freq, inst_phase
    else:
        return env, inst_freq


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
