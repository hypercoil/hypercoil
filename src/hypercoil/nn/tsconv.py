# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Modules supporting convolution of time series data.
"""
from __future__ import annotations
from typing import Callable, Literal, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import equinox as eqx

from ..engine.paramutil import Tensor, _to_jax_array
from ..functional import basisconv2d, polyconv2d, tsconv2d


class TimeSeriesConv2D(eqx.Module):
    """
    Time series convolution.
    """

    weight: Tensor
    bias: Optional[Tensor]
    padding: Optional[
        Union[Literal['initial', 'final'], Sequence[Tuple[int, int]]]
    ]
    in_channels: int
    out_channels: int

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        memory: int = 3,
        kernel_width: int = 1,
        padding: Optional[
            Union[Literal['initial', 'final'], Sequence[Tuple[int, int]]]
        ] = None,
        use_bias: bool = False,
        *,
        key: jax.random.PRNGKey,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        if padding == 'initial' or padding == 'final':
            kernel_duration = memory + 1
        else:
            kernel_duration = memory * 2 + 1
        wkey, bkey = jax.random.split(key, 2)
        lim = 1 / jnp.sqrt(self.in_channels * kernel_width * kernel_duration)
        self.weight = jax.random.uniform(
            key=wkey,
            shape=(
                out_channels,
                self.in_channels,
                kernel_width,
                kernel_duration,
            ),
            minval=-lim,
            maxval=lim,
        )
        if use_bias:
            self.bias = jax.random.uniform(
                key=bkey,
                shape=(self.out_channels,),
                minval=-lim,
                maxval=lim,
            )
        else:
            self.bias = None

    def __call__(
        self,
        input: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        weight = _to_jax_array(self.weight)
        bias = self.bias
        if bias is not None:
            bias = _to_jax_array(bias)
        return tsconv2d(
            X=input,
            weight=weight,
            bias=bias,
            padding=self.padding,
        )


class PolyConv2D(TimeSeriesConv2D):
    r"""
    2D convolution over a polynomial expansion of an input signal.

    In a degree-K polynomial convolution, each input channel is mapped
    across K channels, and the ith channel holds the input raised to the ith
    power. The ith input channel of the kernel thus sees the input raised to
    the ith power.

    :Dimension: **Input :** :math:`(N, *, P, O)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, P denotes number of data channels
                    or variables, O denotes number of time points or
                    observations per channel.
                **Output :** :math:`(N, *, C_{out}, P, O)`
                    :math:`C_{out}` denotes number of output channels.

    Parameters
    ----------
    degree
        Maximum degree K of the polynomial expansion.
    out_channels
        Number of output channels, :math:`C_{out}`.
    memory
        Number of previous observations seen by the kernel.
    kernel_width
        Number of adjoining variables seen by the kernel at once. Unless the
        variables are ordered and evenly sampled, use 1 (the same kernel for
        every variable) or P (a separate kernel for each variable).
    padding
        ``'initial'`` or ``'final'`` makes the kernel causal (memory only)
        and pads the start or end of the series; a sequence of
        ``(before, after)`` pairs pads each of the last two axes explicitly.
        None pads the time axis symmetrically, by half the kernel duration on
        each side.
    use_bias
        Add a learnable bias for each output channel.
    include_const
        Include a constant term in the expansion. Prefer ``use_bias``, which
        is more efficient and behaves better at the edges.
    key
        Pseudo-random number generator key. Weights and bias are initialised
        from :math:`\mathcal{U}(-l, l)`, with :math:`l` the inverse square
        root of the kernel's fan-in.
    """
    include_const: bool

    def __init__(
        self,
        degree: int = 2,
        out_channels: int = 1,
        memory: int = 3,
        kernel_width: int = 1,
        padding: Optional[
            Union[Literal['initial', 'final'], Sequence[Tuple[int, int]]]
        ] = None,
        use_bias: bool = False,
        include_const: bool = False,
        *,
        key: jax.random.PRNGKey,
    ):
        in_channels = degree + include_const
        self.include_const = include_const
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            memory=memory,
            kernel_width=kernel_width,
            padding=padding,
            use_bias=use_bias,
            key=key,
        )

    def __call__(
        self,
        input: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        weight = _to_jax_array(self.weight)
        bias = self.bias
        if bias is not None:
            bias = _to_jax_array(bias)
        return polyconv2d(
            X=input,
            weight=weight,
            bias=bias,
            include_const=self.include_const,
            padding=self.padding,
        )


class BasisConv2D(TimeSeriesConv2D):
    r"""
    2D convolution over a basis expansion of an input signal.
    """
    basis_functions: Sequence[Callable]
    include_const: bool

    def __init__(
        self,
        basis_functions: Sequence[Callable],
        out_channels: int = 1,
        memory: int = 3,
        kernel_width: int = 1,
        padding: Optional[
            Union[Literal['initial', 'final'], Sequence[Tuple[int, int]]]
        ] = None,
        use_bias: bool = False,
        include_const: bool = False,
        *,
        key: jax.random.PRNGKey,
    ):
        in_channels = len(basis_functions) + include_const
        self.basis_functions = basis_functions
        self.include_const = include_const
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            memory=memory,
            kernel_width=kernel_width,
            padding=padding,
            use_bias=use_bias,
            key=key,
        )

    def __call__(
        self,
        input: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        weight = _to_jax_array(self.weight)
        bias = self.bias
        if bias is not None:
            bias = _to_jax_array(bias)
        return basisconv2d(
            X=input,
            weight=weight,
            bias=bias,
            basis_functions=self.basis_functions,
            include_const=self.include_const,
            padding=self.padding,
        )
