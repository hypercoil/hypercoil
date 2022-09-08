# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Modules supporting convolution of time series data.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable, Literal, Optional, Sequence, Tuple, Union
import equinox as eqx
from ..engine.paramutil import Tensor, _to_jax_array
from ..functional import polyconv2d, tsconv2d, basisconv2d


class TimeSeriesConv2D(eqx.Module):
    """
    Time series convolution.
    """
    weight: Tensor
    bias: Optional[Tensor]
    padding: Optional[Union[Literal['initial', 'final'],
                      Sequence[Tuple[int, int]]]]
    in_channels: int
    out_channels: int

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        memory: int = 3,
        kernel_width: int = 1,
        padding: Optional[Union[Literal['initial', 'final'],
                          Sequence[Tuple[int, int]]]] = None,
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
        lim = 1 / jnp.sqrt(
            self.in_channels * kernel_width * kernel_duration)
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

    In a degree-K polynomial convolution, each channel of the input dataset is
    mapped across K channels, and raised to the ith power at the ith channel.
    The convolution kernel's ith input channel thus views the input dataset
    raised to the ith power.

    :Dimension: **Input :** :math:`(N, *, P, O)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, P denotes number of data channels
                    or variables, O denotes number of time points or
                    observations per channel.
                **Output :** :math:`(N, *, C_{out}, P, O)`
                    :math:`C_{out}` denotes number of output channels.

    Parameters
    ----------
    degree : int (default 2)
        Maximum degree of the polynomial expansion.
    out_channels : int (default 1)
        Number of output channels produced by the convolution.
    memory : int (default 3)
        Kernel memory. The number of previous observations viewable by the
        kernel.
    kernel_width : int (default 1)
        Number of adjoining variables simultaneously viewed by the kernel.
        Unless the variables are ordered and evenly sampled, this should either
        be 1 or P. Setting this equal to 1 applies the same kernel to all
        variables, while setting it equal to P applies a unique kernel for
        each variable.
    padding : int or None (default None)
        Number of zero-padding frames added to both sides of the input.
    bias : bool (default False)
        Indicates that a learnable bias should be added channel-wise to the
        output.
    include_const : bool (default False)
        Indicates that a constant term should be included in the polynomial
        expansion. This is almost equivalent to `bias`, and it is advised to
        use `bias` instead because it both is more efficient and exhibits more
        appropriate edge behaviours.

    Attributes
    ----------
    weight : Tensor
        Learnable kernel weights for polynomial convolution.
    bias : Tensor
        Learnable bias for polynomial convolution.
    """
    include_const: bool

    def __init__(
        self,
        degree: int = 2,
        out_channels: int = 1,
        memory: int = 3,
        kernel_width: int = 1,
        padding: Optional[Union[Literal['initial', 'final'],
                          Sequence[Tuple[int, int]]]] = None,
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
            padding=self.padding
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
        padding: Optional[Union[Literal['initial', 'final'],
                          Sequence[Tuple[int, int]]]] = None,
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
            padding=self.padding
        )
