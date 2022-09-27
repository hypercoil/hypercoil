# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Convolve the signal via multiplication in the Fourier domain.
"""
import math
import jax.numpy as jnp
from typing import Callable, Optional, Sequence, Tuple
from .cov import corr
from .utils import complex_decompose
from ..engine import NestedDocParse, Tensor, orient_and_conform


def document_frequency_filter(f: Callable) -> Callable:
    freqfilter_dim_spec = r"""
    :Dimension: **Input :** :math:`(N, *, obs)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, obs denotes number of observations
                    per data channel.
                **Weight :** :math:`(*, \left\lfloor \frac{obs}{2} \right\rfloor + 1)`
                    As above.
                **Output :** :math:`(N, *, 2 \left\lfloor \frac{obs}{2} \right\rfloor )`
                    As above."""
    freqfilter_param_spec = """
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
        Any additional parameters provided will be passed to ``jnp.fft.rfft``
        and ``jnp.fft.irfft``."""
    freqfilter_return_spec = """
    Returns
    -------
    Tensor
        Original time series filtered via multiplication in the frequency
        domain."""
    fmt = NestedDocParse(
        freqfilter_dim_spec=freqfilter_dim_spec,
        freqfilter_param_spec=freqfilter_param_spec,
        freqfilter_return_spec=freqfilter_return_spec,
    )
    f.__doc__ = f.__doc__.format_map(fmt)
    return f


def document_analytic_signal(f: Callable) -> Callable:
    analytic_signal_base_spec = """
    X : tensor
        Input tensor.
    axis : int (default -1)
        Axis along which the transform is applied.
    n : int (default None)
        Number of frequency components; dimension of the Fourier transform.
        This defaults to the size of the input along the transform axis."""
    analytic_signal_sampling_freq = """
    fs : float
        Sampling frequency."""
    analytic_signal_period = """
    period : float (default ``2 * pi``)
        Range over which the signal wraps. (See ``jax.numpy.unwrap``.)"""
    analytic_signal_return_spec = """
    Returns
    -------
    Tensor
        Output tensor.
    """
    analytic_signal_see_also = """
    See also
    --------
    :func:`analytic_signal`
    :func:`hilbert_transform`
    :func:`envelope`
    :func:`instantaneous_phase`
    :func:`instantaneous_frequency`
    :func:`env_inst`"""
    fmt = NestedDocParse(
        analytic_signal_base_spec=analytic_signal_base_spec,
        analytic_signal_sampling_freq=analytic_signal_sampling_freq,
        analytic_signal_period=analytic_signal_period,
        analytic_signal_return_spec=analytic_signal_return_spec,
        analytic_signal_see_also=analytic_signal_see_also
    )
    f.__doc__ = f.__doc__.format_map(fmt)
    return f


@document_frequency_filter
def product_filter(
    X: Tensor,
    weight: Tensor,
    **params
) -> Tensor:
    """
    Convolve a multivariate signal via multiplication in the frequency domain.

    .. note::

        For a filter that is guaranteed to be zero-phase even when the weight
        tensor is not strictly real-valued, use :func:`product_filtfilt`.
    \
    {freqfilter_dim_spec}
    \
    {freqfilter_param_spec}
    \
    {freqfilter_return_spec}
    """
    n = X.shape[-1]
    Xf = jnp.fft.rfft(X, n=n, **params)
    Xf_filt = weight * Xf
    return jnp.fft.irfft(Xf_filt, n=n, **params)


@document_frequency_filter
def product_filtfilt(
    X: Tensor,
    weight: Tensor,
    **params
) -> Tensor:
    """
    Perform zero-phase digital filtering of a signal via multiplication in the
    frequency domain.

    This function operates by first filtering the data and then filtering a
    time-reversed copy of the filtered data again. Note that the effect on the
    amplitude is quadratic in the filter weight.

    .. note::

        If the ``weight`` argument is strictly real, then the filter has no
        phase delay component and it could make sense to simply use
        :func:`product_filter` depending on the context.
    \
    {freqfilter_dim_spec}
    \
    {freqfilter_param_spec}
    \
    {freqfilter_return_spec}
    """
    X_filt = product_filter(X, weight, **params)
    out = product_filter(jnp.flip(X_filt,-1), weight, **params)
    return jnp.flip(out, -1)


def unwrap(
    phase: Tensor,
    axis: int = -1,
    discont: Optional[float] = None,
    period: float = (2 * math.pi)
) -> Tensor:
    r"""
    Unwrap tensor values, replacing large deltas with their complement.

    .. warning::
        This function is retained for backwards compatibility. In all new
        code, use :func:`jax.numpy.unwrap` instead.

    The unwrapping procedure first computes the difference between each pair
    of contiguous values along the specified tensor axis. For each difference
    that is greater than the maximum specified discontinuity (and half the
    period), the corresponding array value is replaced by its complement with
    respect to the period.

    The default case (period of :math:`2\pi`, maximum discontinuity
    :math:`\pi`) corresponds to unwrapping a radian phase such that adjacent
    differences in the phase tensor obtain a maximum value of :math:`\pi`.
    This is achieved by adding :math:`2 k \pi` for an appropriate value of k.

    This mostly follows the implementation in ``numpy``.

    Parameters
    ----------
    phase : tensor
        Tensor containing phases, or other values to be unwrapped. Currently,
        tensors should be cast to some floating-point type before this
        operation.
    axis : int (default -1)
        Axis along which the maximum discontinuity is not be be exceeded after
        unwrapping.
    discont : float (default ``period / 2``)
        Maximum discontinuity between continuous tensor entries along the
        specified ``axis``. Note that this value can in effect be no smaller
        than ``period / 2``.
    period : float (default ``(2 * pi)``)
        Size of the range over which the input tensor wraps.

    Returns
    -------
    tensor
        Tensor containing unwrapped values.
    """
    phase = jnp.asarray(phase)
    dd = jnp.diff(phase, axis=axis)
    half_period = period / 2
    if discont is None:
        discont = half_period

    slice1 = [slice(None, None)] * phase.ndim
    slice1[axis] = slice(1, None)
    slice1 = tuple(slice1)

    interval_high = half_period
    interval_low = -interval_high
    ddmod = (dd - interval_low) % period + interval_low
    ddmod = jnp.where(
        (ddmod == interval_low) & (dd > 0),
        interval_high,
        ddmod
    )
    phase_correct = ddmod - dd
    phase_correct = jnp.where(
        jnp.abs(dd) < discont, 0.0, phase_correct
    )
    unwrapped_phase = phase.at[slice1].set(
        phase[slice1] + jnp.cumsum(phase_correct, axis=axis))
    return unwrapped_phase


@document_analytic_signal
def analytic_signal(
    X: Tensor,
    axis: int = -1,
    n: Optional[int] = None
) -> Tensor:
    """
    Compute the analytic signal.

    The analytic signal is a helical representation of the input signal in
    the complex plane. Its real and imaginary parts are related by the
    Hilbert transform.  Its properties can be used to quickly derive measures
    such as a signal envelope and instantaneous measures of frequency and
    phase.

    Parameters
    ----------\
    {analytic_signal_base_spec}
    \
    {analytic_signal_return_spec}
    \
    {analytic_signal_see_also}
    """
    if jnp.iscomplexobj(X):
        raise ValueError(
            'Input for analytic signal must be strictly real')

    Xf = jnp.fft.fft(X, n=n, axis=axis)
    n = n or X.shape[axis]
    h = jnp.zeros(n)

    #TODO: don't like this assignment implementation or the conditionals
    if n % 2 == 0:
        h = h.at[0].set(1)
        h = h.at[n // 2].set(1)
        h = h.at[1:(n // 2)].set(2)
    else:
        h = h.at[0].set(1)
        h = h.at[1:((n + 1) // 2)].set(2)

    if Xf.ndim >= 1:
        h = orient_and_conform(h, axis=axis, reference=Xf)
    return jnp.fft.ifft(Xf * h, axis=axis)


@document_analytic_signal
def hilbert_transform(
    X: Tensor,
    axis: int = -1,
    n: Optional[int] = None
) -> Tensor:
    """
    Hilbert transform of an input signal.

    Parameters
    ----------\
    {analytic_signal_base_spec}
    \
    {analytic_signal_return_spec}
    \
    {analytic_signal_see_also}
    """
    return analytic_signal(X=X, axis=axis, n=n).imag


@document_analytic_signal
def envelope(
    X: Tensor,
    axis: int = -1,
    n: Optional[int] = None
) -> Tensor:
    """
    Envelope of a signal, computed via the analytic signal.

    .. note::
        If you require the instantaneous phase or frequency in addition to the
        envelope, :func:`env_inst` will be more efficient.

    Parameters
    ----------\
    {analytic_signal_base_spec}
    \
    {analytic_signal_return_spec}
    \
    {analytic_signal_see_also}
    """
    return jnp.abs(analytic_signal(X=X, axis=axis, n=n))


@document_analytic_signal
def instantaneous_phase(
    X: Tensor,
    axis: int = -1,
    n: Optional[int] = None,
    period: float = (2 * math.pi)
) -> Tensor:
    """
    Instantaneous phase of a signal, computed via the analytic signal.

    .. note::
        If you require the envelope or instantaneous frequency in addition to
        the instantaneous phase, :func:`env_inst` will be more efficient.\
    {analytic_signal_base_spec}\
    {analytic_signal_period}\
    \
    {analytic_signal_return_spec}
    \
    {analytic_signal_see_also}
    """
    return jnp.unwrap(
        jnp.angle(analytic_signal(X=X, axis=axis, n=n)),
        axis=axis,
        period=period
    )


def instantaneous_frequency(
    X: Tensor,
    axis: int = -1,
    n : Optional[int] = None,
    fs: float = 1,
    period: float = (2 * math.pi)
) -> Tensor:
    """
    Instantaneous frequency of a signal, computed via the analytic signal.

    .. note::
        If you require the envelope or instantaneous phase in addition to the
        the instantaneous frequency, :func:`env_inst` will be more efficient.

    Parameters
    ----------\
    {analytic_signal_base_spec}\
    {analytic_signal_period}\
    {analytic_signal_sampling_frequency}
    \
    {analytic_signal_return_spec}
    \
    {analytic_signal_see_also}
    """
    inst_phase = instantaneous_phase(X=X, axis=axis, n=n, period=period)
    return fs * jnp.diff(inst_phase, axis=axis) / period


def env_inst(
    X: Tensor,
    axis: int = -1,
    n: Optional[int] = None,
    fs: float = 1,
    period: float = (2 * math.pi)
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute the analytic signal, and then decompose it into the envelope and
    instantaneous phase and frequency.

    Parameters
    ----------\
    {analytic_signal_base_spec}\
    {analytic_signal_period}\
    {analytic_signal_sampling_frequency}
    \
    {analytic_signal_return_spec}
    \
    {analytic_signal_see_also}
    """
    Xa = analytic_signal(X=X, axis=axis, n=n)
    env = jnp.abs(Xa)
    inst_phase = jnp.unwrap(jnp.angle(Xa), axis=axis, period=period)
    inst_freq = fs * jnp.diff(inst_phase, axis=axis) / period
    return env, inst_freq, inst_phase


#TODO: marking this as an experimental function
def ampl_phase_corr(
    X: Tensor,
    weight: Tensor = None,
    corr_axes: Sequence[int] = (0,),
    cov: Callable = corr,
    **params
) -> Tensor:
    """
    Covariance among frequency bins, amplitude and phase. To run it across the
    batch and region axes, use
    ``ampl_phase_corr(X, weight=None, corr_axes=(0, -2))``
    Be advised: there is no interesting structure here.
    """
    n = X.shape[-1]
    Xf = jnp.fft.rfft(X, n=n, **params)
    ampl, phase = complex_decompose(Xf)
    axes = [True for _ in ampl.shape]
    for ax in corr_axes:
        axes[ax] = False
    shape = [e for i, e in enumerate(ampl.shape) if axes[i]]
    new_axes = [i - len(corr_axes) for i in range(len(corr_axes))]
    ampl = jnp.moveaxis(ampl, corr_axes, new_axes).reshape(*shape, -1)
    phase = jnp.moveaxis(phase, corr_axes, new_axes).reshape(*shape, -1)
    return cov(ampl, weight=weight), cov(phase, weight=weight)
