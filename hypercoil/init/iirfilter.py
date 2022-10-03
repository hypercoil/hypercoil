# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Tools for initialising parameters for an IIR filter layer.
"""
from __future__ import annotations
from typing import Literal, Optional, Tuple, Union

import jax.numpy as jnp
import equinox as eqx
from numpy.random import uniform
from scipy import signal

from ..engine import Tensor


# TODO: mark this as experimental until the IIR filter module is properly
#       differentiable.
class IIRFilterSpec(eqx.Module):
    """
    Specification for filter coefficients for recursive IIR filter classes.

    Parameters
    ----------
    Wn : float or tuple(float, float)
        Critical or cutoff frequency. If this is a band-pass filter, then this
        should be a tuple, with the first entry specifying the high-pass cutoff
        and the second entry specifying the low-pass frequency. This should be
        specified relative to the Nyquist frequency if ``fs`` is not provided,
        and should be in the same units as ``fs`` if it is provided.
    N : int or Tensor (default 1)
        Filter order.
    ftype : one of (``'butter'``, ``'cheby1'``, ``'cheby2'``, ``'ellip'``, ``'bessel'``, ``'kuznetsov'``)
        Filter class to initialise: Butterworth, Chebyshev I, Chebyshev II,
        elliptic, or Bessel-Thompson.

        .. note::
            To initialise an ideal filter, use a
            :doc:`frequency product filter <hypercoil.init.freqfilter>`
            instead.
    btype : ``'bandpass'`` (default) or ``'bandstop'`` or ``'lowpass'`` or ``'highpass'``
        Filter pass-band to emulate: low-pass, high-pass, or band-pass. The
        interpretation of the critical frequency changes depending on the
        filter type.
    fs : float or None (default None)
        Sampling frequency.
    rp : float (default 0.1)
        Pass-band ripple. Used only for Chebyshev I and elliptic filters.
    rs : float (default 20)
        Stop-band ripple. Used only for Chebyshev II and elliptic filters.
    norm : ``'phase'`` or ``'mag'`` or ``'delay'`` (default ``'phase'``)
        Critical frequency normalisation. Consult the ``scipy.signal.bessel``
        documentation for details.
    """

    Wn: Union[float, Tuple[float, float]]
    N: int
    ftype: Literal[
        "butter",
        "cheby1",
        "cheby2",
        "ellip",
        "bessel",
    ] = "butter"
    btype: Literal["bandpass", "bandstop", "lowpass", "highpass"] = "bandpass"
    fs: Optional[float] = None
    rp: float = 0.1
    rs: float = 20
    norm: Literal["phase", "mag", "delay"] = "phase"
    coefs: Tensor

    def __init__(
        self,
        Wn: Union[float, Tuple[float, float]],
        N: int,
        ftype: Literal[
            "butter",
            "cheby1",
            "cheby2",
            "ellip",
            "bessel",
        ] = "butter",
        btype: Literal[
            "bandpass", "bandstop", "lowpass", "highpass"
        ] = "bandpass",
        fs: Optional[float] = None,
        rp: float = 0.1,
        rs: float = 20,
        norm: Literal["phase", "mag", "delay"] = "phase",
    ):
        super().__init__()
        self.Wn = Wn
        self.N = N
        self.ftype = ftype
        self.btype = btype
        self.fs = fs
        self.rp = rp
        self.rs = rs
        self.norm = norm
        self.initialise_coefs()

    def initialise_coefs(self):
        if self.ftype == "butter":
            iirfilter = signal.butter
            filter_params = {}
        elif self.ftype == "cheby1":
            iirfilter = signal.cheby1
            filter_params = {"rp": self.rp}
        elif self.ftype == "cheby2":
            iirfilter = signal.cheby2
            filter_params = {"rs": self.rs}
        elif self.ftype == "ellip":
            iirfilter = signal.ellip
            filter_params = {"rp": self.rp, "rs": self.rs}
        elif self.ftype == "bessel":
            iirfilter = signal.bessel
            filter_params = {"norm": self.norm}
        elif self.ftype == "kuznetsov":
            self.coefs = kuznetsov_init(N=self.N, btype=self.btype)
            return
        else:
            raise ValueError(f"Unrecognised filter type : {self.ftype}")
        self.coefs = iirfilter_coefs(
            iirfilter=iirfilter,
            N=self.N,
            Wn=self.Wn,
            btype=self.btype,
            fs=self.fs,
            filter_params=filter_params,
        )


def iirfilter_coefs(
    iirfilter,
    N,
    Wn,
    btype="bandpass",
    fs=None,
    filter_params=None,
) -> Tuple[Tensor, Tensor]:
    filter_params = filter_params or {}
    N = jnp.atleast_1d(N).astype(int)
    Wn = jnp.atleast_1d(Wn).astype(float)
    if btype in ("bandpass", "bandstop") and Wn.ndim < 2:
        Wn = Wn.reshape(-1, 2)
    return tuple(
        iirfilter(N=n, Wn=wn, btype=btype, fs=fs, **filter_params)
        for n, wn in zip(N, Wn)
    )


# TODO: this is not correctly implemented. See
# https://dafx2020.mdw.ac.at/proceedings/papers/DAFx2020_paper_52.pdf
def kuznetsov_init(N, btype="bandpass"):
    multiplier = 1
    if btype in ("bandpass", "bandstop"):
        multiplier = 2
    # val, _ = torch.sort(torch.abs(X))
    b = uniform(-1, 1, N * multiplier + 1)
    a = uniform(-0.5, 0.5, N * multiplier + 1)
    a[0] = 1
    return b, a
