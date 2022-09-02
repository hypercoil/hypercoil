# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Tools for initialising parameters to emulate or replicate the transfer
function of a filter in the frequency domain.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from dataclasses import field
from functools import partial
from typing import (
    Callable, Dict, Literal, Optional, Sequence, Tuple, Type, Union)
from .base import Initialiser, MappedInitialiser, retrieve_parameter
from .mapparam import MappedParameter
from ..functional.utils import PyTree, Tensor, complex_recompose


def document_frequency_spectrum(func):
    iir_warning = """
    .. danger::

        Note that this provides **only an emulation** for IIR filters. Thus,
        for instance, using the ``butterworth`` filter initialisation will not
        in any sense initialise a Butterworth filter. It will return a
        spectrum that approximately emulates the action of a Butterworth
        filter in the frequency domain. In practice, the results are not even
        close to a true IIR filter. For a true differentiable IIR filter, use
        the
        :doc:`IIRFilter <hypercoil.init.iirfilter>`
        class instead (when it's operational)."""
    base_param_spec = """
    N : int or tuple of int
        Filter order. If this is a tuple, then a separate filter will be
        created for each entry in the tuple. Wn must be shaped to match or of
        length 1.
    Wn : float or tuple(float, float) or tuple thereof
        Critical or cutoff frequency. If this is a band-pass filter, then this
        should be a tuple, with the first entry specifying the high-pass
        cutoff and the second entry specifying the low-pass frequency. This
        should be specified relative to the Nyquist frequency if `fs` is not
        provided, and should be in the same units as `fs` if it is provided.
        To create multiple filters, specify a tuple (of floats or tuples)
        containing the critical frequencies for each filter in a single entry.
        N must be shaped to match or of length 1.
    worN : int
        Number of frequency bins to include in the computed spectrum.
    btype : 'lowpass', 'highpass', or 'bandpass' (default 'bandpass')
        Filter type to emulate: low-pass, high-pass, or band-pass. The
        interpretation of the critical frequency changes depending on the
        filter type.
    fs : float or None (default None)
        Sampling frequency.
    """
    rp_param_spec = """
    rp : float
        Pass-band ripple parameter. Used only for Chebyshev I and elliptic
        filter response emulation."""
    rs_param_spec = """
    rs : float
        Stop-band ripple parameter. Used only for Chebyshev II and elliptic
        filter response emulation."""
    norm_param_spec = """
    norm : ``'phase'``, ``'delay'``, or ``'amplitude'``
        Critical frequency normalisation. Used only for Bessel-Thompson
        filter response emulation. Consult the ``scipy.signal.bessel``
        documentation for details."""
    randn_param_spec = """
    ampl_loc : float
        Mean of the normal distribution used to initialise the amplitudes.
        Used only for random initialisation.
    ampl_scale : float
        Standard deviation of the normal distribution used to initialise the
        amplitudes. Used only for random initialisation.
    phase_loc : float
        Mean of the normal distribution used to initialise the phases. Used
        only for random initialisation.
    phase_scale : float
        Standard deviation of the normal distribution used to initialise the
        phases. Used only for random initialisation.
    """
    func.__doc__ = func.__doc__.format(
        iir_warning=iir_warning,
        base_param_spec=base_param_spec,
        rp_param_spec=rp_param_spec,
        rs_param_spec=rs_param_spec,
        norm_param_spec=norm_param_spec,
        randn_param_spec=randn_param_spec
    )
    return func


def filter_spectrum(
    filter: Callable,
    N: Sequence[int],
    Wn: Sequence[Union[float, Tuple[float, float]]],
    worN: int,
    btype: Literal['lowpass', 'highpass', 'bandpass'] = 'bandpass',
    fs: Optional[float] = None,
    filter_params: Optional[dict] = None,
    spectrum_params: Optional[dict] = None
) -> Tensor:
    """
    Transfer function for an IIR filter, obtained via import from scipy.
    \
    {iir_warning}

    Parameters
    ----------
    filter : callable
        `scipy.signal` filter function corresponding to the filter to be
        estimated, for instance `butter` for a Butterworth filter.\
    {base_param_spec}
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
    from scipy.signal import freqz
    N = jnp.array(N).astype(int)
    Wn = jnp.array(Wn).astype(float)
    if btype in ('bandpass', 'bandstop') and Wn.ndim < 2:
        Wn = Wn.reshape(1, 2)
    vals = [
        filter(N=n, Wn=wn, btype=btype, fs=fs, **filter_params)
        for n, wn in zip(N, Wn)
    ]
    fs = fs or 2 * jnp.pi
    vals = [
        freqz(b, a, worN=worN, fs=fs, include_nyquist=True, **spectrum_params)
        for b, a in vals
    ]
    return jnp.stack([v for _, v in vals])


@document_frequency_spectrum
def butterworth_spectrum(
    *,
    N: Sequence[int],
    Wn: Sequence[Union[float, Tuple[float, float]]],
    worN: int,
    btype: Literal['lowpass', 'highpass', 'bandpass'] = 'bandpass',
    fs: Optional[float] = None,
    **params
) -> Tensor:
    """
    Butterworth filter's transfer function obtained via import from scipy.
    \
    {iir_warning}

    Parameters
    ----------\
    {base_param_spec}

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


@document_frequency_spectrum
def chebyshev1_spectrum(
    *,
    N: Sequence[int],
    Wn: Sequence[Union[float, Tuple[float, float]]],
    worN: int,
    btype: Literal['lowpass', 'highpass', 'bandpass'] = 'bandpass',
    fs: Optional[float] = None,
    rp: float,
    **params):
    """
    Chebyshev I filter's transfer function obtained via import from scipy.
    \
    {iir_warning}

    Parameters
    ----------\
    {base_param_spec}\
    {rp_param_spec}

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


@document_frequency_spectrum
def chebyshev2_spectrum(
    *,
    N: Sequence[int],
    Wn: Sequence[Union[float, Tuple[float, float]]],
    worN: int,
    btype: Literal['lowpass', 'highpass', 'bandpass'] = 'bandpass',
    fs: Optional[float] = None,
    rs: float,
    **params
) -> Tensor:
    """
    Chebyshev II filter's transfer function obtained via import from scipy.
    \
    {iir_warning}

    Parameters
    ----------\
    {base_param_spec}\
    {rs_param_spec}

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


@document_frequency_spectrum
def elliptic_spectrum(
    *,
    N: Sequence[int],
    Wn: Sequence[Union[float, Tuple[float, float]]],
    worN: int,
    btype: Literal['lowpass', 'highpass', 'bandpass'] = 'bandpass',
    fs: Optional[float] = None,
    rp: float,
    rs: float,
    **params
) -> Tensor:
    """
    Elliptic filter's transfer function obtained via import from scipy.
    \
    {iir_warning}

    Parameters
    ----------\
    {base_param_spec}\
    {rp_param_spec}\
    {rs_param_spec}

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


@document_frequency_spectrum
def bessel_spectrum(
    *,
    N: Sequence[int],
    Wn: Sequence[Union[float, Tuple[float, float]]],
    worN: int,
    btype: Literal['lowpass', 'highpass', 'bandpass'] = 'bandpass',
    fs: Optional[float] = None,
    norm: Literal['mag', 'phase', 'delay', 'amplitude'] = 'phase',
    **params
) -> Tensor:
    """
    Bessel-Thompson filter's transfer function obtained via import from scipy.
    \
    {iir_warning}

    Parameters
    ----------\
    {base_param_spec}\
    {norm_param_spec}

    Returns
    -------
    out : Tensor
        The specified elliptic transfer function.
    """
    from scipy.signal import bessel
    if norm == 'amplitude': norm = 'mag'
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


@document_frequency_spectrum
def ideal_spectrum(
    *,
    Wn: Sequence[Union[float, Tuple[float, float]]],
    worN: int,
    btype: Literal['lowpass', 'highpass', 'bandpass'] = 'bandpass',
    fs: Optional[float] = None,
    **params
) -> Tensor:
    """
    Ideal filter transfer function.

    Note that the exact specified cutoff frequencies are permitted to pass.

    Parameters
    ----------\
    {base_param_spec}

    Returns
    -------
    out : Tensor
        The specified ideal transfer function.
    """
    Wn = jnp.array(Wn)
    if fs is not None:
        Wn = 2 * Wn / fs
    frequencies = jnp.linspace(0, 1, worN)
    if btype == 'lowpass':
        response = frequencies <= Wn[..., None]
    elif btype == 'highpass':
        response = frequencies >= Wn[..., None]
    elif btype == 'bandpass':
        response_hp = frequencies >= Wn[:, 0][..., None]
        response_lp = frequencies <= Wn[:, 1][..., None]
        response = response_hp * response_lp
    elif btype == 'bandstop':
        response_hp = frequencies <= Wn[:, 0][..., None]
        response_lp = frequencies >= Wn[:, 1][..., None]
        response = response_hp + response_lp
    return response.astype(jnp.float32)


@document_frequency_spectrum
def randn_spectrum(
    *,
    worN: int,
    key: 'jax.random.PRNGKey',
    n_filters: int = 1,
    ampl_loc: float = 0.5,
    ampl_scale: float = 0.2,
    phase_loc: float = 0.,
    phase_scale: float = 0.2,
    **params,
) -> Tensor:
    """
    Transfer function drawn randomly from a normal distribution.

    Parameters
    ----------
    worN : int
        Number of frequencies at which to evaluate the transfer function.
    n_filters : int, optional
        Number of filters to generate. Default: 1.\
    {randn_param_spec}

    Returns
    -------
    out : Tensor
        The specified random transfer function.
    """
    ampl = jax.random.normal(
        key=key, shape=(n_filters, worN)) * ampl_scale + ampl_loc
    phase = jax.random.normal(
        key=key, shape=(n_filters, worN)) * phase_scale + phase_loc
    return complex_recompose(ampl, phase)


class _FreqFilterSpecDefaults(eqx.Module):
    """
    Dataclass default fields for the ``FreqFilterSpec`` class.
    """

    Wn: Optional[Union[
        Tuple[float, ...], Tuple[Tuple[float, float], ...]
    ]] = None
    N: Tuple[int] = (1,)
    ftype: Literal[
        'ideal', 'butter', 'cheby1', 'cheby2',
        'ellip', 'bessel', 'randn'] = 'ideal'
    btype: Literal['lowpass', 'highpass', 'bandpass', 'bandstop'] = 'lowpass'
    fs: Optional[float] = None
    rp: float = 0.1
    rs: float = 0.1
    norm: Literal['phase', 'amplitude', 'delay'] = 'phase'
    ampl_loc: float = 0.5
    ampl_scale: float = 0.1
    phase_loc: float = 0.
    phase_scale: float = 0.02
    clamps: Optional[Sequence[Dict[float, float]]] = None
    bound: float = 3.
    lookup: Dict[str, Callable] = field(default_factory = lambda: {
        'ideal': ideal_spectrum,
        'butter': butterworth_spectrum,
        'cheby1': chebyshev1_spectrum,
        'cheby2': chebyshev2_spectrum,
        'ellip': elliptic_spectrum,
        'bessel': bessel_spectrum,
        'randn': randn_spectrum,
    })


@document_frequency_spectrum
class FreqFilterSpec(_FreqFilterSpecDefaults):
    """
    Specification for an approximate frequency response curve for various
    filter classes.
    \
    {iir_warning}

    Parameters
    ----------\
    ftype : one of (``'butter'``, ``'cheby1'``, ``'cheby2'``, ``'ellip'``, ``'bessel'``, ``'ideal'``, ``'randn'``)
        Filter class to emulate: Butterworth, Chebyshev I, Chebyshev II,
        elliptic, Bessel-Thompson, ideal, or filter weights sampled randomly
        from a normal distribution.

        .. note::
            Because it cannot be overstated: IIR filter spectra (Butterworth,
            Chebyshev, elliptic, Bessel) are strictly emulations and do not
            remotely follow the actual behaviour of the filter. These filters
            are recursive and cannot be implemented as frequency products. By
            contrast, the ideal initialisation is exact.
    {base_param_spec}\
    {rs_param_spec}\
    {rp_param_spec}\
    {norm_param_spec}\
    {randn_param_spec}
    clamps : list(dict)
        Frequencies whose responses should be clampable to particular values.
        Each element of the list is a dictionary corresponding to a single
        filter; the list should therefore have a length of ``F``. Each
        key/value pair in each dictionary should correspond to a frequency
        (relative to Nyquist or ``fs`` if provided) and the clampable response
        at that frequency. For instance {{0.1: 0, 0.5: 1}} enables clamping of
        the frequency bin closest to 0.1 to 0 (full stop) and the bin closest
        to 0.5 to 1 (full pass). Note that the clamp must be applied using the
        ``get_clamps`` method.
    """
    n_filters: int

    def __init__(
        self,
        *,
        Wn: Optional[Union[
            float, Tuple[float, float],
            Tuple[float, ...], Tuple[Tuple[float, float], ...]
        ]] = None,
        N: Union[int, Tuple[int, ...]] = 1,
        ftype: Literal[
            'ideal', 'butter', 'cheby1', 'cheby2',
            'ellip', 'bessel', 'randn'] = 'ideal',
        btype: Literal[
            'lowpass', 'highpass', 'bandpass', 'bandstop'] = 'bandpass',
        fs: Optional[float] = None,
        rp: float = 0.1,
        rs: float = 0.1,
        norm: Literal['phase', 'amplitude', 'delay'] = 'phase',
        ampl_loc: float = 0.5,
        ampl_scale: float = 0.1,
        phase_loc: float = 0.,
        phase_scale: float = 0.02,
        clamps: Optional[Sequence[Dict[float, float]]] = None,
        bound: float = 3.,
    ):
        if isinstance(N, int):
            N = (N,)
        if Wn is not None:
            if btype == 'bandpass' or btype == 'bandstop':
                if isinstance(Wn[0], float):
                    Wn = (Wn,)
            else:
                if isinstance(Wn, float):
                    Wn = (Wn,)
            if len(N) == 1 and len(Wn) > 1:
                N = N * len(Wn)
            elif len(N) > 1 and len(Wn) == 1:
                Wn = Wn * len(N)
        if clamps is not None:
            if isinstance(clamps, dict):
                clamps = (clamps,)
            if len(N) > 1 and len(clamps) == 1:
                clamps = clamps * len(N)
        self.n_filters = len(Wn)
        super().__init__(
            Wn=Wn, N=N, ftype=ftype, btype=btype, fs=fs, rp=rp, rs=rs,
            norm=norm, ampl_loc=ampl_loc, ampl_scale=ampl_scale,
            phase_loc=phase_loc, phase_scale=phase_scale, clamps=clamps,
            bound=bound
        )

    def get_clamps(
        self,
        *,
        worN: int,
    ):
        """
        Returns a mask and a set of values that can be used to clamp each
        filter's transfer function at a specified set of frequencies.

        To apply the clamp, use:

        ```jnp.where(points, values, spectrum)```

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
        frequencies = jnp.linspace(0, 1, worN)
        if self.clamps is None or len(self.clamps) == 0:
            null = jnp.zeros((self.n_filters, worN), dtype=bool)
            return null, null
        clamp_data = [
            self._clamp_filter(clamps, frequencies, worN)
            for clamps in self.clamps
        ]
        points, values = zip(*clamp_data)
        return (
            jnp.concatenate(points, axis=0),
            jnp.concatenate(values, axis=0)
        )

    def _clamp_filter(
        self,
        clamps: Dict[float, float],
        frequencies: jnp.ndarray,
        worN: int,
    ):
        fs = self.fs or 1.
        clamp_points = jnp.array(list(clamps.keys())) / fs
        clamp_values = jnp.array(list(clamps.values()))
        dist = jnp.abs(clamp_points[:, None] - frequencies[None, :])
        mask = jax.nn.one_hot(dist.argmin(-1), num_classes=worN)
        clamp_points = mask.sum(0, keepdims=True).astype(bool)
        try:
            assert clamp_points.sum() == len(clamp_values)
        except AssertionError as e:
            e.args += ('Unable to separately resolve clamped frequencies',)
            e.args += ('Increase either the spacing or the number of bins',)
            raise
        clamp_values = jnp.where(
            mask, clamp_values[:, None], 0.).sum(0, keepdims=True)
        return clamp_points, clamp_values

    def initialise_spectrum(
        self,
        *,
        worN: int,
        key: 'jax.random.PRNGKey',
    ) -> Tensor:
        """
        Initialises a frequency spectrum or transfer function approximation
        for the specified filter with ``worN`` frequency bins between 0 and
        Nyquist, inclusive.

        Parameters
        ----------
        worN : int
            Number of frequency bins between 0 and Nyquist, inclusive.

        Returns
        -------
        Tensor
            A tensor of shape ``(F, worN)`` where ``F`` is the number of
            filters.
        """
        return self.lookup[self.ftype](
            key=key,
            worN=worN,
            **self.__dict__,
        )


@document_frequency_spectrum
def freqfilter_init(
    *,
    shape: Tuple[int, ...],
    filter_specs: Sequence[FreqFilterSpec],
    key: 'jax.random.PRNGKey',
) -> Tensor:
    """
    Filter transfer function initialisation.

    Initialise a tensor such that its values follow or approximate the
    transfer function of a specified filter.
    \
    {iir_warning}

    :Dimension: **tensor :** :math:`(*, F, N)`
                    F denotes the total number of filters to initialise from
                    the provided specs, and N denotes the number of frequency
                    bins.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the tensor to initialise.
    filter_specs : list(FreqFilterSpec)
        A list of filter specifications implemented as :class:`FreqFilterSpec`
        objects.

    Returns
    -------
    Tensor
        A tensor of shape ``shape`` containing the transfer functions of the
        specified filters.
    """
    worN = shape[-1]
    spectra = [
        fspec.initialise_spectrum(worN=worN, key=key)
        for fspec in filter_specs
    ]
    spectra = jnp.concatenate(spectra, axis=-2)
    shape = list(shape)
    shape[-2] = spectra.shape[-2]
    return jnp.broadcast_to(spectra, shape)


def clamp_init(
    *,
    shape: Tuple[int, ...],
    filter_specs: Sequence[FreqFilterSpec],
    key: Optional['jax.random.PRNGKey'] = None,
):
    """
    Filter clamp initialisation.

    Initialise two tensors such that the first masks the points of a filter's
    transfer function to be clamped and the second contains the clamping
    values.

    :Dimension: **points_tensor :** :math:`(*, F, N)`
                    F denotes the total number of filters to initialise from
                    the provided specs, and N denotes the number of frequency
                    bins.
                **values_tensor :** :math:`(*, F, N)`
                    As above

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the tensors to initialise.
    filter_specs : list(FreqFilterSpec)
        A list of filter specifications implemented as :class:`FreqFilterSpec`
        objects.

    Returns
    -------
    points_tensor : Tensor
        A boolean tensor of shape ``shape`` that masks the points of a filter's
        transfer function to be clamped.
    values_tensor : Tensor
        A tensor of shape ``shape`` that contains the clamping values.
    """
    worN = shape[-1]
    clamps = [
        fspec.get_clamps(worN=worN) for fspec in filter_specs
    ]
    points, values = zip(*clamps)
    return jnp.concatenate(points, axis=0), jnp.concatenate(values, axis=0)


class FreqFilterInitialiser(MappedInitialiser):
    """
    Initialises a frequency filter with the specified parameters.

    See :class:`FreqFilterSpec`,
    :func:`freqfilter_init`,
    :func:`clamp_init`, and
    :class:`MappedInitialiser` for argument details.
    """

    filter_specs: Sequence[FreqFilterSpec]

    def __init__(
        self,
        filter_specs: Sequence[FreqFilterSpec],
        mapper: Optional[Type[MappedParameter]] = None,
    ):
        self.filter_specs = filter_specs
        super().__init__(mapper=mapper)

    def __call__(
        self,
        model: PyTree,
        *,
        param_name: str = "weight",
        key: jax.random.PRNGKey,
        clamp: bool = False,
        **params,
    ):
        if clamp:
            shape = retrieve_parameter(model, param_name=param_name)[0].shape
            init_fn = clamp_init
            key = None
        else:
            shape = retrieve_parameter(model, param_name=param_name).shape
            init_fn = freqfilter_init
        return self._init(
            shape=shape, key=key, init_fn=init_fn, **params,
        )

    def _init(
        self,
        shape=Tuple[int, ...],
        key='jax.random.PRNGKey',
        init_fn: Callable[..., Tensor] = freqfilter_init,
    ):
        return init_fn(
            shape=shape, filter_specs=self.filter_specs, key=key
        )

    @staticmethod
    def _init_impl(
        init: Initialiser,
        model: PyTree,
        param_name: str,
        clamp_name: str,
        key: Optional[jax.random.PRNGKey],
        **params
    ) -> PyTree:
        model = eqx.tree_at(
            partial(retrieve_parameter, param_name=param_name),
            model,
            replace=init(
                model=model, param_name=param_name, key=key)
        )
        if clamp_name is not None:
            model = eqx.tree_at(
                partial(retrieve_parameter, param_name=clamp_name),
                model,
                replace=init(
                    model=model, param_name=clamp_name, key=key, clamp=True)
            )
        if init.mapper is None:
            return model
        return init.mapper.map(model=model, param_name=param_name, **params)

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Optional[Type[MappedParameter]] = None,
        filter_specs: Sequence[FreqFilterSpec],
        param_name: str = 'weight',
        clamp_name: Optional[str] = None,
        key: 'jax.random.PRNGKey',
        **params,
    ):
        init = cls(
            mapper=mapper,
            filter_specs=filter_specs,
        )
        return cls._init_impl(
            init=init, model=model,
            param_name=param_name,
            clamp_name=clamp_name,
            key=key, **params,
        )


class _FreqFilterSpec(object):
    def __init__(self, Wn=None, N=1, ftype='butter', btype='bandpass', fs=None,
                 rp=0.1, rs=20, norm='phase', ampl_loc=0.5, ampl_scale=0.1,
                 phase_loc=0, phase_scale=0.02, clamps=None, bound=3):
        raise NotImplementedError()

def freqfilter_init_(tensor, filter_specs, domain=None):
    raise NotImplementedError()

def clamp_init_(points_tensor, values_tensor, filter_specs):
    raise NotImplementedError()
