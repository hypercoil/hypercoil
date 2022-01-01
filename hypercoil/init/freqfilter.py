# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Frequency filter initialisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tools for initialising parameters to emulate the transfer function of a filter.
"""
import torch
import math
from ..functional.domain import AmplitudeAtanh, Clip
from ..functional.activation import complex_recompose


class FreqFilterSpec(object):
    """
    Specification for an IIR or ideal filter.

    Dimension
    ---------
    - N : :math:`(F)`
      F denotes the total number of filters to initialise.
    - Wn : :math:`(F, 2)` for bandpass or bandstop or :math:`(F)` otherwise

    Parameters
    ----------
    Wn : float or tuple(float, float) or Tensor
        Critical or cutoff frequency. If this is a band-pass filter, then this
        should be a tuple, with the first entry specifying the high-pass cutoff
        and the second entry specifying the low-pass frequency. This should be
        specified relative to the Nyquist frequency if `fs` is not provided,
        and should be in the same units as `fs` if it is provided. To create
        multiple filters, specify a tensor containing the critical frequencies
        for each filter in a single row.
    N : int or Tensor (default 1)
        Filter order. If this is a tensor, then a separate filter will be
        created for each entry in the tensor. Wn must be shaped to match. Not
        used for ideal filters.
    ftype : one of ('butter', 'cheby1', 'cheby2', 'ellip',
                    'bessel', 'ideal', 'randn')
        Filter class to emulate: Butterworth, Chebyshev I, Chebyshev II,
        elliptic, Bessel-Thompson, ideal, or filter weights sampled randomly
        from a normal distribution.
    btype : 'bandpass' (default) or 'bandstop' or 'lowpass' or 'highpass'
        Filter pass-band to emulate: low-pass, high-pass, or band-pass. The
        interpretation of the critical frequency changes depending on the
        filter type.
    fs : float or None (default None)
        Sampling frequency.
    rp : float (default 0.1)
        Pass-band ripple. Used only for Chebyshev I and elliptic filters.
    rs : float (default 20)
        Stop-band ripple. Used only for Chebyshev II and elliptic filters.
    norm : 'phase' or 'mag' or 'delay' (default 'phase')
        Critical frequency normalisation. Consult the `scipy.signal.bessel`
        documentation for details.
    ampl_loc : float
        If randomly sampling filter weights from a normal distribution, this
        parameter specifies the mean of the amplitude distribution.
    ampl_scale: float
        If randomly sampling filter weights from a normal distribution, this
        parameter specifies the standard deviation of the amplitude
        distribution. (Note that there is currently no support for a covariant
        phase-amplitude distribution.)
    phase_loc : float
        If randomly sampling filter weights from a normal distribution, this
        parameter specifies the mean of the phase distribution.
    phase_scale: float
        If randomly sampling filter weights from a normal distribution, this
        parameter specifies the standard deviation of the phase distribution.
        (Note that there is currently no support for a covariant phase-
        amplitude distribution.)
    clamps : list(dict)
        Frequencies whose responses should be clampable to particular values.
        Each element of the list is a dictionary corresponding to a single
        filter; the list should therefore have a length of `F`. Each key/value
        pair in each dictionary should correspond to a frequency (relative to
        Nyquist or `fs` if provided) and the clampable response at that
        frequency. For instance {0.1: 0, 0.5: 1} enables clamping of the
        frequency bin closest to 0.1 to 0 (full stop) and the bin closest to
        0.5 to 1 (full pass). Note that the clamp must be applied using the
        `get_clamps` method.
    bound : float (default 3)
        Maximum tolerable amplitude in the transfer function. Any values in
        excess will be adjusted according to the `ood` option when a spectrum
        is initialised.

    Attributes
    ----------
    spectrum : Tensor
        Populated when the `initialise_spectrum` method is called.
    """
    def __init__(self, Wn=None, N=1, ftype='butter', btype='bandpass', fs=None,
                 rp=0.1, rs=20, norm='phase', ampl_loc=0.5, ampl_scale=0.1,
                 phase_loc=0, phase_scale=0.02, clamps=None, bound=3):
        if Wn is not None:
            N = _ensure_tensor(N)
            Wn = _ensure_tensor(Wn)
            if btype in ('bandpass', 'bandstop') and Wn.ndim < 2:
                Wn = Wn.view(-1, 2)
            if N.size(0) == 1 and Wn.size(0) > N.size(0):
                N = N.repeat(Wn.size(0), 1)
            elif Wn.size(0) == 1 and Wn.size(0) < N.size(0):
                Wn = Wn.repeat(N.size(0), 1)
        else:
            assert(ftype == 'randn')
        self.N = N
        self.Wn = Wn
        self.ftype = ftype
        self.btype = btype
        self.fs = fs
        self.rp = rp
        self.rs = rs
        self.norm = norm
        self.ampl_loc = ampl_loc
        self.ampl_scale = ampl_scale
        self.phase_loc = phase_loc
        self.phase_scale = phase_scale
        self.clamps = clamps or []
        self.bound = bound
        try:
            self.n_filters = Wn.size(0)
        except AttributeError:
            self.n_filters = 1

    def initialise_spectrum(self, worN, domain=None):
        """
        Initialises a frequency spectrum or transfer function for the specified
        filter with `worN` frequency bins between 0 and Nyquist, inclusive.

        Parameters
        ----------
        worN : int
            Number of frequency bins between 0 and Nyquist, inclusive.
        domain : Domain object (default AmplitudeAtanh)
            A domain object from `hypercoil.functional.domain`, used to specify
            the domain of the output spectrum. An `Identity` object yields the
            raw transfer function, while an `AmplitudeAtanh` object transforms
            the amplitudes of each bin by the inverse tanh (atanh) function.
            This transformation can be useful if the transfer function will be
            used as a learnable parameter whose amplitude will be transformed
            by the tanh function, thereby constraining it to [0, 1) and
            preventing explosive gain.

        Returns
        -------
        None: the `spectrum` attribute is populated instead.
        """
        domain = domain or AmplitudeAtanh(handler=Clip())
        if self.ftype == 'butter':
            self.spectrum = butterworth_spectrum(
                N=self.N, Wn=self.Wn, btype=self.btype, worN=worN, fs=self.fs)
        elif self.ftype == 'cheby1':
            self.spectrum = chebyshev1_spectrum(
                N=self.N, Wn=self.Wn, rp=self.rp, btype=self.btype,
                worN=worN, fs=self.fs)
        elif self.ftype == 'cheby2':
            self.spectrum = chebyshev2_spectrum(
                N=self.N, Wn=self.Wn, rs=self.rs, btype=self.btype,
                worN=worN, fs=self.fs)
        elif self.ftype == 'ellip':
            self.spectrum = elliptic_spectrum(
                N=self.N, Wn=self.Wn, rp=self.rp, rs=self.rs,
                btype=self.btype, worN=worN, fs=self.fs)
        elif self.ftype == 'bessel':
            self.spectrum = bessel_spectrum(
                N=self.N, Wn=self.Wn, norm=self.norm,
                btype=self.btype, worN=worN, fs=self.fs)
        elif self.ftype == 'ideal':
            self.spectrum = ideal_spectrum(
                Wn=self.Wn, btype=self.btype, worN=worN, fs=self.fs)
        elif self.ftype == 'randn':
            self.spectrum = randn_spectrum(
                worN=worN, ampl_loc=self.ampl_loc, ampl_scale=self.ampl_scale,
                phase_loc=self.phase_loc, phase_scale=self.phase_scale,
                n_filters=self.n_filters)
        self.spectrum = domain.preimage(self.spectrum)

    def get_clamps(self, worN):
        """
        Returns a mask and a set of values that can be used to clamp each
        filter's transfer function at a specified set of frequencies.

        To apply the clamp, use:

        spectrum[points] = values

        Parameters
        ----------
        worN : int
            Number of frequency bins between 0 and Nyquist, inclusive.

        Returns
        -------
        points : Tensor
            Boolean mask indicating the frequency bins that are clampable.
        values : Tensor
            Values to which the response function is clampable at the specified
            frequencies.
        """
        frequencies = torch.linspace(0, 1, worN)
        points, values = [], []
        if len(self.clamps) == 0:
            return torch.zeros((self.n_filters, worN)).bool(), torch.Tensor([])
        for clamps in self.clamps:
            mask, vals = self._filter_clamps(clamps, frequencies, worN)
            points.append(mask)
            values.append(vals)
        if len(points) > 1:
            return torch.cat(points, 0), torch.cat(values)
        return mask, vals

    def _filter_clamps(self, clamps, frequencies, worN):
        clamp_points = torch.Tensor(list(clamps.keys()))
        clamp_values = torch.Tensor(list(clamps.values()))
        if len(clamp_values) == 0:
            return torch.zeros((1, worN)).bool(), torch.Tensor([])
        fs = self.fs or 1
        clamp_points /= fs
        dist = torch.abs(clamp_points.view(-1, 1) - frequencies)
        mask = torch.eye(worN)[dist.argmin(-1)]
        clamp_points = mask.sum(0).view(1, -1).bool()
        try:
            assert clamp_points.sum() == len(clamp_values)
        except AssertionError as e:
            e.args += ('Unable to separately resolve clamped frequencies',)
            e.args += ('Increase either the spacing or the number of bins',)
            raise
        return clamp_points, clamp_values

    def __repr__(self):
        s = (f'FreqFilterSpec(ftype={self.ftype}, n_filters={self.n_filters})')
        return s


def freqfilter_init_(tensor, filter_specs, domain=None):
    """
    IIR filter-like transfer function initialisation.

    Initialise a tensor such that its values follow the transfer function of
    an IIR or ideal filter. For IIR filters, the transfer function is computed
    as a frequency response curve in scipy.

    Dimension
    ---------
    - tensor : :math:`(*, F, N)`
      F denotes the total number of filters to initialise from the provided
      specs, and N denotes the number of frequency bins.

    Parameters
    ----------
    tensor : Tensor
        Tensor to initialise in-place. The import will include only the real
        part (and will therefore be incorrect for most filters) if the provided
        tensor does not have a complex datatype. Note that even if the transfer
        function is strictly real, the gradient will almost certainly not be
        and it is therefore critical that this tensor allow complex values.
    filter_specs : list(FreqFilterSpec)
        A list of filter specifications implemented as `FreqFilterSpec` objects
        (`hypercoil.init.FreqFilterSpec`).
    domain : Domain object (default AmplitudeAtanh)
        A domain object from `hypercoil.functional.domain`, used to specify
        the domain of the output spectrum. An `Identity` object yields the
        raw transfer function, while an `AmplitudeAtanh` object transforms
        the amplitudes of each bin by the inverse tanh (atanh) function.
        This transformation can be useful if the initialised tensor will be
        used as a learnable parameter whose amplitude will be transformed
        by the tanh function, thereby constraining it to [0, 1) and
        preventing explosive gain.

    Returns
    -------
    None. The input tensor is initialised in-place.
    """
    rg = tensor.requires_grad
    tensor.requires_grad = False
    worN = tensor.size(-1)
    for fspec in filter_specs:
        fspec.initialise_spectrum(worN, domain)
    spectra = torch.cat([fspec.spectrum for fspec in filter_specs])
    tensor[:] = spectra
    tensor.requires_grad = rg


def clamp_init_(points_tensor, values_tensor, filter_specs):
    """
    Filter clamp initialisation.

    Initialise two tensors such that the first masks the points of a filter's
    transfer function to be clamped and the second contains the clamping
    values.

    Dimension
    ---------
    - points_tensor : :math:`(*, F, N)`
      F denotes the total number of filters to initialise from the provided
      specs, and N denotes the number of frequency bins.
    - values_tensor : :math:`(K)`
      K denotes the total number of values to be clamped.

    Parameters
    ----------
    points_tensor : Tensor
        Mask tensor to initialise in-place. This should be of dtype
        `torch.bool` or similar so that it can operate as a mask.
    values_tensor : Tensor
        Tensor containing clamping values to initialise in-place.
    filter_specs : list(FreqFilterSpec)
        A list of filter specifications implemented as `FreqFilterSpec` objects
        (`hypercoil.init.FreqFilterSpec`).

    Returns
    -------
    None. The input tensors are initialised in-place.
    """
    rgp = points_tensor.requires_grad
    rgv = values_tensor.requires_grad
    points_tensor.requires_grad = False
    values_tensor.requires_grad = False
    worN = points_tensor.size(-1)
    points, values = [], []
    for fspec in filter_specs:
        mask, vals = fspec.get_clamps(worN)
        points.append(mask)
        values.append(vals)
    if len(points) > 1:
        points, values = torch.cat(points, 0), torch.cat(values)
    else:
        points, values = points[0], values[0]
    points_tensor[:] = points
    values_tensor[:] = values
    points_tensor.requires_grad = rgp
    values_tensor.requires_grad = rgv


def butterworth_spectrum(N, Wn, worN, btype='bandpass', fs=None):
    """
    Butterworth filter's transfer function obtained via import from scipy.

    Dimension
    ---------
    - N : :math:`(F)`
      F denotes the total number of filter to initialise.
    - Wn : :math:`(F, 2)` for bandpass or bandstop or :math:`(F)` otherwise

    Parameters
    ----------
    N : int or Tensor
        Filter order. If this is a tensor, then a separate filter will be
        created for each entry in the tensor. Wn must be shaped to match.
    Wn : float or tuple(float, float) or Tensor
        Critical or cutoff frequency. If this is a band-pass filter, then this
        should be a tuple, with the first entry specifying the high-pass cutoff
        and the second entry specifying the low-pass frequency. This should be
        specified relative to the Nyquist frequency if `fs` is not provided,
        and should be in the same units as `fs` if it is provided. To create
        multiple filters, specify a tensor containing the critical frequencies
        for each filter in a single row.
    worN : int
        Number of frequency bins to include in the computed spectrum.
    btype : 'lowpass', 'highpass', or 'bandpass' (default 'bandpass')
        Filter type to emulate: low-pass, high-pass, or band-pass. The
        interpretation of the critical frequency changes depending on the
        filter type.
    fs : float or None (default None)
        Sampling frequency.

    Returns
    -------
    out : Tensor
        The specified Butterworth transfer function.
    """
    from scipy.signal import butter
    filter_params, spectrum_params = {}, {}
    return filter_spectrum(
        filter=butter,
        N=N, Wn=Wn, worN=worN,
        btype=btype, fs=fs,
        filter_params=filter_params,
        spectrum_params=spectrum_params)


def chebyshev1_spectrum(N, Wn, worN, rp, btype='bandpass', fs=None):
    """
    Chebyshev I filter's transfer function obtained via import from scipy.

    Dimension
    ---------
    - N : :math:`(F)`
      F denotes the total number of filter to initialise.
    - Wn : :math:`(F, 2)` for bandpass or bandstop or :math:`(F)` otherwise

    Parameters
    ----------
    N : int or Tensor
        Filter order. If this is a tensor, then a separate filter will be
        created for each entry in the tensor. Wn must be shaped to match.
    Wn : float or tuple(float, float) or Tensor
        Critical or cutoff frequency. If this is a band-pass filter, then this
        should be a tuple, with the first entry specifying the high-pass cutoff
        and the second entry specifying the low-pass frequency. This should be
        specified relative to the Nyquist frequency if `fs` is not provided,
        and should be in the same units as `fs` if it is provided. To create
        multiple filters, specify a tensor containing the critical frequencies
        for each filter in a single row.
    worN : int
        Number of frequency bins to include in the computed spectrum.
    rp : float
        Pass-band ripple parameter.
    btype : 'lowpass', 'highpass', or 'bandpass' (default 'bandpass')
        Filter type to emulate: low-pass, high-pass, or band-pass. The
        interpretation of the critical frequency changes depending on the
        filter type.
    fs : float or None (default None)
        Sampling frequency.

    Returns
    -------
    out : Tensor
        The specified Chebyshev I transfer function.
    """
    from scipy.signal import cheby1
    filter_params = {
        'rp': rp
    }
    spectrum_params = {}
    return filter_spectrum(
        filter=cheby1,
        N=N, Wn=Wn, worN=worN,
        btype=btype, fs=fs,
        filter_params=filter_params,
        spectrum_params=spectrum_params)


def chebyshev2_spectrum(N, Wn, worN, rs, btype='bandpass', fs=None):
    """
    Chebyshev II filter's transfer function obtained via import from scipy.

    Dimension
    ---------
    - N : :math:`(F)`
      F denotes the total number of filter to initialise.
    - Wn : :math:`(F, 2)` for bandpass or bandstop or :math:`(F)` otherwise

    Parameters
    ----------
    N : int or Tensor
        Filter order. If this is a tensor, then a separate filter will be
        created for each entry in the tensor. Wn must be shaped to match.
    Wn : float or tuple(float, float) or Tensor
        Critical or cutoff frequency. If this is a band-pass filter, then this
        should be a tuple, with the first entry specifying the high-pass cutoff
        and the second entry specifying the low-pass frequency. This should be
        specified relative to the Nyquist frequency if `fs` is not provided,
        and should be in the same units as `fs` if it is provided. To create
        multiple filters, specify a tensor containing the critical frequencies
        for each filter in a single row.
    worN : int
        Number of frequency bins to include in the computed spectrum.
    rs : float
        Stop-band ripple parameter.
    btype : 'lowpass', 'highpass', or 'bandpass' (default 'bandpass')
        Filter type to emulate: low-pass, high-pass, or band-pass. The
        interpretation of the critical frequency changes depending on the
        filter type.
    fs : float or None (default None)
        Sampling frequency.

    Returns
    -------
    out : Tensor
        The specified Chebyshev II transfer function.
    """
    from scipy.signal import cheby2
    filter_params = {
        'rs': rs
    }
    spectrum_params = {}
    return filter_spectrum(
        filter=cheby2,
        N=N, Wn=Wn, worN=worN,
        btype=btype, fs=fs,
        filter_params=filter_params,
        spectrum_params=spectrum_params)


def elliptic_spectrum(N, Wn, worN, rp, rs, btype='bandpass', fs=None):
    """
    Elliptic filter's transfer function obtained via import from scipy.

    Dimension
    ---------
    - N : :math:`(F)`
      F denotes the total number of filter to initialise.
    - Wn : :math:`(F, 2)` for bandpass or bandstop or :math:`(F)` otherwise

    Parameters
    ----------
    N : int or Tensor
        Filter order. If this is a tensor, then a separate filter will be
        created for each entry in the tensor. Wn must be shaped to match.
    Wn : float or tuple(float, float) or Tensor
        Critical or cutoff frequency. If this is a band-pass filter, then this
        should be a tuple, with the first entry specifying the high-pass cutoff
        and the second entry specifying the low-pass frequency. This should be
        specified relative to the Nyquist frequency if `fs` is not provided,
        and should be in the same units as `fs` if it is provided. To create
        multiple filters, specify a tensor containing the critical frequencies
        for each filter in a single row.
    worN : int
        Number of frequency bins to include in the computed spectrum.
    rp : float
        Pass-band ripple parameter.
    rs : float
        Stop-band ripple parameter.
    btype : 'lowpass', 'highpass', or 'bandpass' (default 'bandpass')
        Filter type to emulate: low-pass, high-pass, or band-pass. The
        interpretation of the critical frequency changes depending on the
        filter type.
    fs : float or None (default None)
        Sampling frequency.

    Returns
    -------
    out : Tensor
        The specified elliptic transfer function.
    """
    from scipy.signal import ellip
    filter_params = {
        'rp': rp,
        'rs': rs
    }
    spectrum_params = {}
    return filter_spectrum(
        filter=ellip,
        N=N, Wn=Wn, worN=worN,
        btype=btype, fs=fs,
        filter_params=filter_params,
        spectrum_params=spectrum_params)


def bessel_spectrum(N, Wn, worN, norm='phase', btype='bandpass', fs=None):
    """
    Bessel-Thompson filter's transfer function obtained via import from scipy.

    Dimension
    ---------
    - N : :math:`(F)`
      F denotes the total number of filter to initialise.
    - Wn : :math:`(F, 2)` for bandpass or bandstop or :math:`(F)` otherwise

    Parameters
    ----------
    N : int or Tensor
        Filter order. If this is a tensor, then a separate filter will be
        created for each entry in the tensor. Wn must be shaped to match.
    Wn : float or tuple(float, float) or Tensor
        Critical or cutoff frequency. If this is a band-pass filter, then this
        should be a tuple, with the first entry specifying the high-pass cutoff
        and the second entry specifying the low-pass frequency. This should be
        specified relative to the Nyquist frequency if `fs` is not provided,
        and should be in the same units as `fs` if it is provided. To create
        multiple filters, specify a tensor containing the critical frequencies
        for each filter in a single row.
    worN : int
        Number of frequency bins to include in the computed spectrum.
    norm : 'phase', 'delay' or 'mag'
        Critical frequency normalisation. Consult the `scipy.signal.bessel`
        documentation for details.
    btype : 'lowpass', 'highpass', or 'bandpass' (default 'bandpass')
        Filter type to emulate: low-pass, high-pass, or band-pass. The
        interpretation of the critical frequency changes depending on the
        filter type.
    fs : float or None (default None)
        Sampling frequency.

    Returns
    -------
    out : Tensor
        The specified elliptic transfer function.
    """
    from scipy.signal import bessel
    filter_params = {
        'norm': norm
    }
    spectrum_params = {}
    return filter_spectrum(
        filter=bessel,
        N=N, Wn=Wn, worN=worN,
        btype=btype, fs=fs,
        filter_params=filter_params,
        spectrum_params=spectrum_params)


def ideal_spectrum(Wn, worN, btype='bandpass', fs=None):
    """
    Ideal filter transfer function.

    Note that the exact specified cutoff frequencies are permitted to pass.

    Dimension
    ---------
    - Wn : :math:`(F, 2)` for bandpass or bandstop or :math:`(F)` otherwise

    Parameters
    ----------
    Wn : float or tuple(float, float) or Tensor
        Critical or cutoff frequency. If this is a band-pass filter, then this
        should be a tuple, with the first entry specifying the high-pass cutoff
        and the second entry specifying the low-pass frequency. This should be
        specified relative to the Nyquist frequency if `fs` is not provided,
        and should be in the same units as `fs` if it is provided. To create
        multiple filters, specify a tensor containing the critical frequencies
        for each filter in a single row.
    worN : int
        Number of frequency bins to include in the computed spectrum.
    btype : 'lowpass', 'highpass', or 'bandpass' (default 'bandpass')
        Filter type to emulate: low-pass, high-pass, or band-pass. The
        interpretation of the critical frequency changes depending on the
        filter type.
    fs : float or None (default None)
        Sampling frequency.

    Returns
    -------
    out : Tensor
        The specified ideal transfer function.
    """
    Wn = _ensure_tensor(Wn)
    if btype in ('bandpass', 'bandstop') and Wn.ndim < 2:
        Wn = Wn.view(-1, 2)
    if fs is not None:
        Wn = 2 * Wn / fs
    frequencies = torch.linspace(0, 1, worN)
    if btype == 'lowpass':
        response = frequencies <= Wn.view(-1, 1)
    elif btype == 'highpass':
        response = frequencies >= Wn.view(-1, 1)
    elif btype == 'bandpass':
        response_hp = frequencies >= Wn[:, 0].view(-1, 1)
        response_lp = frequencies <= Wn[:, 1].view(-1, 1)
        response = response_hp * response_lp
    elif btype == 'bandstop':
        response_hp = frequencies <= Wn[:, 0].view(-1, 1)
        response_lp = frequencies >= Wn[:, 1].view(-1, 1)
        response = response_hp + response_lp
    return response.float()


def randn_spectrum(worN, n_filters=1, ampl_loc=0.5, ampl_scale=0.2,
                   phase_loc=0, phase_scale=0.3):
    ampl = torch.randn(size=(n_filters, worN)) * ampl_scale + ampl_loc
    phase = torch.randn(size=(n_filters, worN)) * phase_scale + phase_loc
    return complex_recompose(ampl, phase)


def filter_spectrum(filter, N, Wn, worN, btype='bandpass', fs=None,
                    filter_params=None, spectrum_params=None):
    """
    Transfer function for an IIR filter, obtained via import from scipy.

    Dimension
    ---------
    - N : :math:`(F)`
      F denotes the total number of filter to initialise.
    - Wn : :math:`(F, 2)` for bandpass or bandstop or :math:`(F)` otherwise

    Parameters
    ----------
    filter : callable
        `scipy.signal` filter function corresponding to the filter to be
        estimated, for instance `butter` for a Butterworth filter.
    N : int or Tensor
        Filter order. If this is a tensor, then a separate filter will be
        created for each entry in the tensor. Wn must be shaped to match.
    Wn : float or tuple(float, float) or Tensor
        Critical or cutoff frequency. If this is a band-pass filter, then this
        should be a tuple, with the first entry specifying the high-pass cutoff
        and the second entry specifying the low-pass frequency. This should be
        specified relative to the Nyquist frequency if `fs` is not provided,
        and should be in the same units as `fs` if it is provided. To create
        multiple filters, specify a tensor containing the critical frequencies
        for each filter in a single row.
    worN : int
        Number of frequency bins to include in the computed spectrum.
    btype : 'lowpass', 'highpass', or 'bandpass' (default 'bandpass')
        Filter type to emulate: low-pass, high-pass, or band-pass. The
        interpretation of the critical frequency changes depending on the
        filter type.
    fs : float or None (default None)
        Sampling frequency.
    filter_params : dict
        Additional parameters to pass to the `filter` callable other than
        those passed directly to this function (for instance, pass- and
        stop-band ripples).
    spectrum_params : dict
        Additional parameters to pass to the `freqz` function that computes
        the frequency response spectrum other than those passed directly to
        this function.

    Returns
    -------
    out : Tensor
        The specified filter's transfer function.
    """
    import numpy as np
    from scipy.signal import freqz
    N = _ensure_ndarray(N).astype(int)
    Wn = _ensure_ndarray(Wn)
    if btype in ('bandpass', 'bandstop') and Wn.ndim < 2:
        Wn = Wn.reshape(-1, 2)
    vals = [
        filter(N=n, Wn=wn, btype=btype, fs=fs, **filter_params)
        for n, wn in zip(N, Wn)
    ]
    fs = fs or 2 * math.pi
    vals = [
        freqz(b, a, worN=worN, fs=fs, include_nyquist=True, **spectrum_params)
        for b, a in vals
    ]
    vals = np.stack([v for _, v in vals])
    return _import_complex_numpy(vals)


def _ensure_ndarray(obj):
    """
    Ensure that the object is an iterable ndarray with dimension greater than
    or equal to 1. Another function we'd do well to get rid of in the future.
    """
    import numpy as np
    try:
        i = iter(obj)
        return np.array(obj)
    except TypeError:
        return np.array([obj])


def _ensure_tensor(obj):
    """
    Ensure that the object is an iterable tensor with dimension greater than
    or equal to 1. Another function we'd do well to get rid of in the future.
    """
    try:
        i = iter(obj)
        return torch.Tensor(obj)
    except TypeError:
        return torch.Tensor([obj])


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
