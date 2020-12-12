# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Butterworth initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~~
Initialise parameters to approximate the attenuation curve of a Butterworth
filter.
"""
import torch
import math


def butterworth_init_(tensor, N, Wn, btype='bandpass', fs=None):
    """
    Butterworth-like attenuation initialisation.

    Initialise a tensor such that its values follow the formula for the
    absolute gain/attenuation curve of a Butterworth filter. This follows the
    formula

    :math:`\frac{1}{\sqrt{1 + \frac{\\omega}{\\omega_C}}}`

    for a low-pass filter and

    :math:`\frac{1}{\sqrt{1 + \frac{\\omega_C}{\\omega}}}`

    for a high-pass filter. For a band-pass filter, the two are multiplied
    together and then scaled so that the maximum value is 1, which is
    absolutely NOT correct. This function will be revisited at some point to
    produce results that are more correct; at this point, the entire system
    will likely require an overhaul for complex-number compatibility.

    Note: this is not the gain curve that exists for a Butterworth filter in
    practice. For instance, this contains only a real component. See
    `butterworth_correct_init_` for a working solution.

    Parameters
    ----------
    tensor : Tensor
        Tensor to initialise in-place.
    N : int
        Filter order.
    Wn : float or tuple(float, float)
        Critical or cutoff frequency. If this is a band-pass filter, then this
        should be a tuple, with the first entry specifying the high-pass cutoff
        and the second entry specifying the low-pass frequency. This should be
        specified relative to the Nyquist frequency if `fs` is not provided,
        and should be in the same units as `fs` if it is provided.
    btype : 'lowpass', 'highpass', or 'bandpass' (default 'bandpass')
        Filter type to emulate: low-pass, high-pass, or band-pass. The
        interpretation of the critical frequency changes depending on the
        filter type.
    fs : float or None (default None)
        Sampling frequency.

    Returns
    -------
    None. The input tensor is initialised in-place.

    See also
    --------
    butterworth_correct_init_: correct, complex-valued initialisation
    """
    rg = tensor.requires_grad
    tensor.requires_grad = False
    if fs is not None:
        Wn = 2 * Wn / fs
    frequencies = torch.linspace(0, 1, tensor.size(-1))
    if btype == 'bandpass':
        hi, lo = Wn
        lovals = 1 / torch.sqrt(1 + (frequencies / lo) ** (2 * N))
        hivals = 1 / torch.sqrt(1 + (hi / frequencies) ** (2 * N))
        vals = lovals * hivals
        vals /= vals.max()
    elif btype == 'lowpass':
        lo = Wn
        vals = 1 / torch.sqrt(1 + (frequencies / lo) ** (2 * N))
    elif btype == 'highpass':
        hi = Wn
        vals = 1 / torch.sqrt(1 + (hi / frequencies) ** (2 * N))
    tensor[:] = vals
    tensor.requires_grad = rg


def butterworth_correct_init_(tensor, N, Wn, btype='bandpass', fs=None):
    """
    An initialisation that matches the Butterworth filter's attenuation
    spectrum, by importing from scipy. The tensor must have a complex datatype
    for the import to succeed.

    Parameters
    ----------
    tensor : Tensor
        Tensor to initialise in-place. The import will include only the real
        part (and will therefore be incorrect) if the provided tensor does not
        have a complex datatype.
    N : int
        Filter order.
    Wn : float or tuple(float, float)
        Critical or cutoff frequency. If this is a band-pass filter, then this
        should be a tuple, with the first entry specifying the high-pass cutoff
        and the second entry specifying the low-pass frequency. This should be
        specified relative to the Nyquist frequency if `fs` is not provided,
        and should be in the same units as `fs` if it is provided.
    btype : 'lowpass', 'highpass', or 'bandpass' (default 'bandpass')
        Filter type to emulate: low-pass, high-pass, or band-pass. The
        interpretation of the critical frequency changes depending on the
        filter type.
    fs : float or None (default None)
        Sampling frequency.

    Returns
    -------
    None. The input tensor is initialised in-place.
    """
    from scipy.signal import butter, freqz
    rg = tensor.requires_grad
    tensor.requires_grad = False
    b, a = butter(N, Wn, btype=btype, fs=fs)
    fs = fs or 2 * math.pi
    vals = freqz(b, a, worN=(tensor.size(-1)), fs=fs, include_nyquist=True)[1]
    tensor[:] = _import_complex_numpy(vals)
    tensor.requires_grad = rg


def _import_complex_numpy(array):
    """
    Hacky import of complex-valued array from numpy into torch. Hopefully this
    can go away in the future. Simply calling torch.Tensor casts the input to
    real, and we would otherwise have to specify a particular precision for the
    import which might not match the precision desired.
    """
    real = torch.Tensor(array.real)
    imag = torch.Tensor(array.imag)
    val = torch.stack([real, imag], -1)
    return torch.view_as_complex(val)
