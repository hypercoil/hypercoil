# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Polynomial convolution
~~~~~~~~~~~~~~~~~~~~~~
Functions supporting polynomial convolution of time series and other data.
"""
import torch


def polyconv2d(X, weight, include_const=False, bias=None,
               padding=None, **params):
    r"""
    Perform convolution using a polynomial channel basis.

    Convolution using a kernel whose ith input channel views the input dataset
    raised to the ith power.

    :Dimension: **Input :** :math:`(N, *, C, obs)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, C denotes number of data channels
                    or variables, obs denotes number of observations per
                    channel.
                **Weight :** :math:`(C_o, K, *, C\ \mathrm{or}\ 1, R)`
                    :math:`C_o` denotes number of output channels, K denotes
                    number of input channels (equivalent to the maximum degree
                    of the polynomial), and R < obs denotes the number of
                    observations viewed at once (i.e., in a single convolutional
                    window)
                **Output :** :math:`(N, C_o, *, C, obs)`
                    As above.

    Parameters
    ----------
    X : Tensor
        Input dataset to be expanded polynomially. The last two dimensions are
        seen by each kernel channel. For time series data, these could be
        variables and sequential observations.
    weight : Tensor
        Polynomial convolution kernel. The first dimension corresponds to the
        output channels and the second to the input channels, each of which
        corresponds to the input X raised to a different power. The final
        dimension corresponds to the number of observations convolved together.
        For a time series, this corresponds to R // 2 past frames, R // 2
        future frames, and the current frame. For a time series, the
        penultimate dimension determines whether the same convolution is
        applied to all variables (if it is 1) or variable-specific convolutions
        can be learned (if it is equal to the number of variables). To permit
        some diversity in kernels while enforcing consistency across variables,
        it is possible to penalise a measure of spread such as the variance
        across the variable axis.
    include_const : bool (default False)
        Indicates that a constant or intercept term corresponding to the zeroth
        power should be included. The first channel of the weight sees the
        constant term.
    bias : Tensor or None
        Bias term for convolution. See `torch.conv2d` documentation for
        details.
    padding : 2-tuple or None
        Padding for convolution, as for `torch.conv2d`. If not explicitly
        specified, this will default to 'time series' padding: no padding in
        the penultimate axis, and R // 2 in the final axis.
    **params
        Additional parameters can be passed to `torch.conv2d`.

    Returns
    -------
    out : Tensor
        Input dataset transformed via polynomial convolution.
    """
    degree = weight.size(1) - include_const
    padding = padding or (0, weight.size(-1) // 2)
    X = polychan(X, degree=degree, include_const=include_const)
    return torch.conv2d(X, weight, bias=bias, padding=padding, **params)


def polychan(X, degree=2, include_const=False):
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

    Dimension
    ---------
    - Input: :math:`(N, *, C, obs)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      C denotes number of data channels or variables, obs denotes number of
      observations
    - Output: :math:`(N, K, *, C, obs)`
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
    if X.dim() > 2:
        pass
    elif X.dim() == 2:
        X = X.view(1, *X.size())
    elif X.dim() == 1:
        X = X.view(1, -1)
    stack = [X]
    for _ in range(degree - 1):
        stack += [stack[-1] * X]
    if include_const:
        stack = [torch.ones_like(X)] + stack
    return torch.stack(stack, 1)
