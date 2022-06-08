# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Additional activation functions for neural network layers.
"""
import torch
from .utils import complex_decompose, complex_recompose


def laplace(input, loc=0, width=1):
    r"""
    Double exponential activation function.

    The double exponential activation function is applied elementwise as

    :math:`e^{\frac{-|x - \mu|}{b}}`

    to inputs x with centre :math:`\mu` and width b. It constrains its outputs
    to the range (0, 1], mapping values closer to its centre to larger outputs.
    It is Lipschitz continuous over the reals and differentiable except at its
    centre.

    :Dimension: As this activation function is applied elementwise, it
                conserves dimension; the output will be of the same shape as
                the input.

    Parameters
    ----------
    input : Tensor
        Tensor to be transformed elementwise by the double exponential
        activation function.
    loc : float or broadcastable Tensor (default 0)
        Centre parameter :math:`\mu` of the double exponential function.
    width : float or broadcastable Tensor (default 1)
        Spread parameter b of the double exponential function.

    Returns
    -------
    out : Tensor
        Transformed input tensor.
    """
    return torch.exp(-torch.abs(input - loc) / width)


def expbarrier(input, barrier=1):
    r"""
    Exponential barrier activation function.

    The exponential barrier activation function is applied elementwise as

    :math:`b \sqrt{1 - \exp{\frac{-|x|}{b^2}}}`

    to inputs x with barrier b. It constrains its outputs to the range [0, b).
    It is Lipschitz continuous over the reals and differentiable except at its
    centre.

    :Dimension: As this activation function is applied elementwise, it
                conserves dimension; the output will be of the same shape as
                the input.

    Parameters
    ----------
    input : Tensor
        Tensor to be transformed elementwise by the exponential barrier
        activation function.
    barrier : float or broadcastable Tensor (default 0)
        Barrier parameter b of the exponential function.

    Returns
    -------
    out : Tensor
        Transformed input tensor.
    """
    ampl = torch.abs(input)
    return barrier * torch.sqrt(1 - torch.exp(-ampl / barrier ** 2))


def amplitude_laplace(input, loc=0, width=1):
    r"""
    Double exponential activation function applied to the amplitude only.

    The amplitude (absolute value) of the input is transformed according to

    :math:`e^{\frac{-|x - \mu|}{b}}`

    while the phase (complex argument) is preserved. This function maps the
    complex plane to the open unit disc: the origin is mapped to the perimeter
    and distant regions of the complex plane are mapped to the origin. The
    function varies quickly near the origin (the region of the plane mapped
    close to the perimeter and small gradient updates could result in large
    changes in output. Furthermore, the function is completely discontinuous
    and undefined at the origin (the direction is ambiguous and any point in
    the circumference is equally valid).

    :Dimension: As this activation function is applied elementwise, it
                conserves dimension; the output will be of the same shape as
                the input.

    Parameters
    ----------
    input : Tensor
        Tensor whose amplitude is to be transformed elementwise by the double
        exponential activation function.
    loc : float or broadcastable Tensor (default 0)
        Centre parameter :math:`\mu` of the double exponential function.
    width : float or broadcastable Tensor (default 1)
        Spread parameter b of the double exponential function.

    Returns
    -------
    out : Tensor
        Transformed input tensor.
    """
    ampl = torch.abs(input - loc)
    fact = torch.exp(-ampl / width) / ampl
    return input * fact


def amplitude_expbarrier(input, barrier=1):
    r"""
    Exponential barrier activation function applied to the amplitude only.

    The amplitude (absolute value) of the input is transformed according to

    :math:`b \sqrt{1 - \exp{\frac{-|x|}{b^2}}}`

    while the phase (complex argument) is preserved. This function maps the
    complex plane to the open unit disc: the origin is mapped to itself
    and distant regions of the complex plane are mapped to the circumference.

    :Dimension: As this activation function is applied elementwise, it
                conserves dimension; the output will be of the same shape as
                the input.

    Parameters
    ----------
    input : Tensor
        Tensor whose amplitude is to be transformed elementwise by the double
        exponential activation function.
    loc : float or broadcastable Tensor (default 0)
        Centre parameter :math:`\mu` of the double exponential function.
    width : float or broadcastable Tensor (default 1)
        Spread parameter b of the double exponential function.

    Returns
    -------
    out : Tensor
        Transformed input tensor.
    """
    ampl, phase = complex_decompose(input)
    xfm = barrier * torch.sqrt(1 - torch.exp(-ampl / barrier ** 2))
    return xfm * torch.exp(phase * 1j)


def amplitude_tanh(input):
    r"""
    Hyperbolic tangent activation function applied to the amplitude only.

    The amplitude (absolute value) of the input is transformed according to

    :math:`\mathrm{tanh} x`

    while the phase (complex argument) is preserved. This function maps the
    complex plane to the open unit disc: the origin is mapped to itself
    and distant regions of the complex plane are mapped to the circumference.

    :Dimension: As this activation function is applied elementwise, it
                conserves dimension; the output will be of the same shape as
                the input.

    Parameters
    ----------
    input : Tensor
        Tensor whose amplitude is to be transformed elementwise by the
        hyperbolic tangent activation function.

    Returns
    -------
    out : Tensor
        Transformed input tensor.
    """
    ampl, phase = complex_decompose(input)
    return complex_recompose(torch.tanh(ampl), phase)


def amplitude_atanh(input):
    r"""
    Inverse hyperbolic tangent (hyperbolic arctangent) activation function
    applied to the amplitude only.

    The amplitude (absolute value) of the input is transformed according to

    :math:`\mathrm{arctanh} x`

    while the phase (complex argument) is preserved. This function maps the
    open unit disc in the complex plane into the entire complex plane: the
    origin is mapped to itself and the circumference is mapped to infinity.

    :Dimension: As this activation function is applied elementwise, it
                conserves dimension; the output will be of the same shape as
                the input.

    Parameters
    ----------
    input : Tensor
        Tensor whose amplitude is to be transformed elementwise by the
        hyperbolic arctangent function.

    Returns
    -------
    out : Tensor
        Transformed input tensor.
    """
    ampl, phase = complex_decompose(input)
    return complex_recompose(torch.atanh(ampl), phase)


def _corrnorm_factor(input):
    factor = torch.diagonal(input, dim1=-2, dim2=-1)
    factor = (-torch.sign(factor) *
              torch.sqrt(torch.abs(factor))).unsqueeze(-1)
    factor = (factor @ factor.transpose(-1, -2) +
              torch.finfo(input.dtype).eps)
    return factor


def corrnorm(input, factor=None, gradpath='both'):
    r"""
    Correlation normalisation activation function.

    Divide each entry :math:`A_{ij}` of the input matrix :math:`A` by the
    product of the signed square roots of the corresponding diagonals:

    :math:`\bar{A}_{ij} = A_{ij} \frac{\mathrm{sgn}(A_{ii} A_{jj})}{\sqrt{A_{ii}}\sqrt{A_{jj}} + \epsilon}`

    This default behaviour, which maps a covariance matrix to a Pearson
    correlation matrix, can be overriden by providing a ``factor`` argument
    (detailed below). This activation function is also similar to a signed
    version of the normalisation operation for a graph Laplacian matrix.

    :Dimension: As this activation function is applied elementwise, it
                conserves dimension; the output will be of the same shape as
                the input.

    Parameters
    ----------
    input : Tensor
        Tensor to be normalised.
    factor : Tensor, iterable(Tensor, Tensor), or None (default None)
        Normalisation factor.

        * If this is not explicitly specified, it follows the default
          behaviour (division by the product of signed square roots.)
        * If this is a tensor, ``input`` is directly divided by the
          provided tensor. This option is provided mostly for compatibility
          with non-square inputs.
        * If this is a pair of tensors, then ``input`` is divided by their
          outer product.
    gradpath : str ``'input'`` or ``'both'`` (default ``'both'``)
        If this is set to ``'input'`` and the default normalisation behaviour
        is used, then gradient will be blocked from flowing backward through
        the computation of the normalisation factor from the input.

    Returns
    -------
    Tensor
        Normalised input.
    """
    if isinstance(factor, torch.Tensor):
        return input / factor
    elif factor is not None:
        factor = (factor[0] @ factor[1].transpose(-1, -2) +
                  torch.finfo(input.dtype).eps)
    elif gradpath == 'input':
        with torch.no_grad():
            factor = _corrnorm_factor(input)
    else:
        factor = _corrnorm_factor(input)
    return input / factor
