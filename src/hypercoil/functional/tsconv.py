# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Functions supporting convolution of time series and other data.
"""
from __future__ import annotations
from typing import (
    Callable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import jax
import jax.numpy as jnp
from equinox.nn._conv import _ntuple

from ..engine import NestedDocParse, Tensor, atleast_4d


torch_dims = {
    0: ('NC', 'OI', 'NC'),
    1: ('NCH', 'OIH', 'NCH'),
    2: ('NCHW', 'OIHW', 'NCHW'),
    3: ('NCHWD', 'OIHWD', 'NCHWD'),
}


def document_time_series_convolution(f: Callable) -> Callable:
    ts_conv_dim = r"""
    :Dimension: **Input :** :math:`(N, *, C, obs)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, C denotes number of data channels
                    or variables, obs denotes number of observations per
                    channel.
                **Weight :** :math:`(C_o, K, *, C\ \mathrm{or}\ 1, R)`
                    :math:`C_o` denotes number of output channels, K denotes
                    number of input channels (equivalent to the maximum degree
                    of the polynomial), and R < obs denotes the number of
                    observations viewed at once (i.e., in a single
                    convolutional window)
                **Output :** :math:`(N, C_o, *, C, obs)`
                    As above."""

    basis_conv_spec = """
    X : Tensor
        Input dataset to the basis functions. The last two dimensions are
        seen by each kernel channel. For time series data, these could be
        variables and sequential observations.
    weight : Tensor
        Basis convolution kernel. The first dimension corresponds to the
        output channels and the second to the input channels, each of which
        corresponds to the evaluation of a basis function over ``X``. The
        final dimension corresponds to the number of observations convolved
        together. For a time series, this corresponds to ``R // 2`` past
        frames, ``R // 2`` future frames, and the current frame. For a time
        series, the penultimate dimension determines whether the same
        convolution is applied to all variables (if it is 1) or
        variable-specific convolutions can be learned (if it is equal to the
        number of variables). To permit some diversity in kernels while
        enforcing consistency across variables, it is possible to penalise
        a measure of spread such as the variance across the variable axis.
    include_const : bool (default False)
        Indicates that a constant or intercept term should be included. The
        first channel of the weight sees the constant term.
    bias : Tensor or None
        Bias term for convolution. See ``torch.conv2d`` documentation for
        details.
    padding : 2-tuple or None
        Padding for convolution, as for ``torch.conv2d``. If not explicitly
        specified, this will default to ``'time series'`` padding: no padding
        in the penultimate axis, and ``R // 2`` in the final axis.
    **params
        Additional parameters can be passed to ``torch.conv2d``."""

    basis_conv_return_spec = """
    Returns
    -------
    Tensor
        Input dataset transformed via basis channel convolution."""

    fmt = NestedDocParse(
        ts_conv_dim=ts_conv_dim,
        basis_conv_spec=basis_conv_spec,
        basis_conv_return_spec=basis_conv_return_spec,
    )
    f.__doc__ = f.__doc__.format_map(fmt)
    return f


def conv(
    input: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    stride: Union[int, Sequence[int]] = 1,
    padding: Union[int, Sequence[Tuple[int, int]]] = 0,
    dilation: Union[int, Sequence[int]] = 1,
) -> Tensor:
    """
    ``torch``-like API for convolution.
    The implementation is basically pilfered from
    https://stackoverflow.com/questions/69571976/ ...
    ... how-to-use-grad-convolution-in-google-jax
    """
    # TODO: Compare against
    # https://github.com/patrick-kidger/equinox/blob/main/equinox/nn/conv.py#L104
    # and reconcile. Pretty sure dilation is not handled correctly here.
    n = len(input.shape) - 2
    parse = _ntuple(n)
    if isinstance(stride, int):
        stride = parse(stride)
    if isinstance(padding, int):
        padding = [(i, i) for i in parse(padding)]
    if isinstance(dilation, int):
        dilation = parse(dilation)
    out = jax.lax.conv_general_dilated(
        lhs=input,
        rhs=weight,
        window_strides=stride,
        padding=padding,
        lhs_dilation=None,
        rhs_dilation=dilation,
        dimension_numbers=torch_dims[n],
        feature_group_count=1,
        batch_group_count=1,
        precision=None,
        preferred_element_type=None,
    )
    if bias is not None:
        return out + bias
    return out


@document_time_series_convolution
def tsconv2d(
    X: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    padding: Optional[
        Union[
            Literal['initial', 'final'],
            Sequence[Tuple[int, int]],
        ]
    ] = None,
    conv_fn: Optional[Callable] = None,
    **params,
) -> Tensor:
    """
    Convolve time series data.

    This is a convenience function for performing convolution over time
    series data. It automatically configures padding so that the convolution
    produces an output with the same number of observations as the input.

    Padding can be applied to the initial or final observations of the input
    dataset, or both.
    \
    {ts_conv_dim}
    """
    X = _configure_input_for_ts_conv(X)
    weight = _configure_weight_for_ts_conv(weight)
    if conv_fn is None:
        conv_fn = conv
    padding = _configure_padding_for_ts_conv(padding, weight)
    return conv_fn(
        input=X,
        weight=weight,
        bias=bias,
        padding=padding,
        **params,
    )


@document_time_series_convolution
def basisconv2d(
    X: Tensor,
    weight: Tensor,
    basis_functions: Sequence[Callable],
    include_const: bool = False,
    bias: Optional[Tensor] = None,
    padding: Optional[Tuple[int, int]] = None,
    conv_fn: Optional[Callable] = None,
    **params,
) -> Tensor:
    """
    Perform convolution using basis function channel mapping.

    Convolution using a kernel whose ith input channel evaluates the ith basis
    function over the input dataset.
    \
    {ts_conv_dim}

    Parameters
    ----------\
    {basis_conv_spec}
    \
    {basis_conv_return_spec}
    """
    weight = _configure_weight_for_ts_conv(weight)
    assert weight.shape[1] - include_const == len(basis_functions)
    padding = _configure_padding_for_ts_conv(padding, weight)
    X = basischan(
        X,
        basis_functions=basis_functions,
        include_const=include_const,
    )
    return tsconv2d(
        X=X,
        weight=weight,
        bias=bias,
        padding=padding,
        conv_fn=conv_fn,
        **params,
    )


@document_time_series_convolution
def polyconv2d(
    X: Tensor,
    weight: Tensor,
    include_const: bool = False,
    bias: Optional[Tensor] = None,
    padding: Optional[
        Union[
            Literal['initial', 'final'],
            Sequence[Tuple[int, int]],
        ]
    ] = None,
    conv_fn: Optional[Callable] = None,
    **params,
) -> Tensor:
    """
    Perform convolution using a polynomial channel basis.

    Convolution using a kernel whose ith input channel views the input dataset
    raised to the ith power.

    .. warning::
        This function automatically creates a channel for each specified
        power. If your input already includes a channel for each power, you
        should use ``tsconv2d`` instead.
    \
    {ts_conv_dim}

    Parameters
    ----------\
    {basis_conv_spec}
    \
    {basis_conv_return_spec}
    """
    weight = _configure_weight_for_ts_conv(weight)
    degree = weight.shape[1] - include_const
    X = polychan(X, degree=degree, include_const=include_const)
    return tsconv2d(
        X=X,
        weight=weight,
        bias=bias,
        padding=padding,
        conv_fn=conv_fn,
        **params,
    )


def _configure_input_for_ts_conv(X: Tensor) -> Tensor:
    return atleast_4d(X)


def _configure_weight_for_ts_conv(weight: Tensor) -> Tensor:
    return atleast_4d(weight)


def _configure_padding_for_ts_conv(
    padding: Union[
        None,
        Literal['initial', 'final'],
        Sequence[Tuple[int, int]],
    ],
    weight: Tensor,
) -> Sequence[Tuple[int, int]]:
    size = weight.shape[-1]
    if padding == 'final':
        padding = ((0, 0), (0, size - 1))
    elif padding == 'initial':
        padding = ((0, 0), (size - 1, 0))
    padding = padding or ((0, 0), (size // 2, size // 2))
    return padding


def basischan(
    X: Tensor,
    basis_functions: List[Callable],
    include_const: bool = False,
) -> Tensor:
    r"""
    Create a channel basis for the data.

    Given K basis functions, single-channel data are mapped across K channels,
    and the ith channel is constituted by evaluating the ith basis function
    over the input data.

    :Dimension: **Input :** :math:`(N, *, C, obs)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, C denotes number of data channels
                    or variables, obs denotes number of observations per
                    channel.
                **Output :** :math:`(N, K, *, C, obs)`
                    K denotes the number of basis functions.

    Parameters
    ----------
    X : Tensor
        Dataset to expand with basis functions. A new channel will be created
        containing the same dataset transformed using each basis function.
    basis_functions : list(callable)
       Functions to use to constitute the basis. Each function's signature
       must map a single input to a single output of the same dimension. Use
       partial functions as appropriate to conform function signatures to this
       requirement.
    include_const : bool (default False)
        Indicates that a constant or intercept term should be included.
    """
    X = _configure_input_for_ts_conv(X)
    if include_const:
        basis_functions = [jnp.ones_like] + list(basis_functions)
    stack = [f(X) for f in basis_functions]
    if X.shape[1] != 1:
        return jnp.stack(stack, 1)
    return jnp.concatenate(stack, 1)


def polychan(
    X: Tensor,
    degree: int = 2,
    include_const: bool = False,
) -> Tensor:
    r"""
    Create a polynomial channel basis for the data.

    Single-channel data are mapped across K channels, and raised to the ith
    power at the ith channel.

    :Dimension: **Input :** :math:`(N, *, C, obs)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, C denotes number of data channels
                    or variables, obs denotes number of observations per
                    channel.
                **Output :** :math:`(N, K, *, C, obs)`
                    K denotes maximum polynomial degree to include

    Parameters
    ----------
    X : Tensor
        Dataset to expand as a polynomial basis. A new channel will be created
        containing the same dataset raised to each power up to the specified
        degree.
    degree : int >= 2 (default 2)
        Maximum polynomial degree to be included in the output basis.
    include_const : bool (default False)
        Indicates that a constant or intercept term corresponding to the zeroth
        power should be included.

    Returns
    -------
    out : Tensor
        Input dataset expanded as a K-channel polynomial basis.
    """
    X = _configure_input_for_ts_conv(X)
    powers = jnp.arange(not include_const, degree + 1)
    return jnp.power(X, powers[..., None, None])
