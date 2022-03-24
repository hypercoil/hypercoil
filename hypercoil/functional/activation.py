# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Activation functions
~~~~~~~~~~~~~~~~~~~~
Additional activation functions for neural network layers.
"""
import torch


def laplace(input, loc=0, width=1):
    r"""
    Double exponential activation function.

    The double exponential activation function is applied elementwise as

    :math:`e^{\frac{-|x - \mu|}{b}}`

    to inputs x with centre :math:`\mu` and width b. It constrains its outputs
    to the range (0, 1], mapping values closer to its centre to larger outputs.
    It is Lipschitz continuous over the reals and differentiable except at its
    centre.

    Dimension
    ---------
    As this activation function is applied elementwise, it conserves dimension;
    the output will be of the same shape as the input.

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

    Dimension
    ---------
    As this activation function is applied elementwise, it conserves dimension;
    the output will be of the same shape as the input.

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

    Dimension
    ---------
    As this activation function is applied elementwise, it conserves dimension;
    the output will be of the same shape as the input.

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

    Dimension
    ---------
    As this activation function is applied elementwise, it conserves dimension;
    the output will be of the same shape as the input.

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

    Dimension
    ---------
    As this activation function is applied elementwise, it conserves dimension;
    the output will be of the same shape as the input.

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

    Dimension
    ---------
    As this activation function is applied elementwise, it conserves dimension;
    the output will be of the same shape as the input.

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


def complex_decompose(complex):
    """
    Decompose a complex-valued tensor into amplitude and phase components.

    Dimension
    ---------
    Each output is of the same shape as the input.

    Parameters
    ----------
    complex : Tensor
        Complex-valued tensor.

    Returns
    -------
    ampl : Tensor
        Amplitude of each entry in the input tensor.
    phase : Tensor
        Phase of each entry in the input tensor, in radians.
    """
    ampl = torch.abs(complex)
    phase = torch.angle(complex)
    return ampl, phase


def complex_recompose(ampl, phase):
    """
    Reconstitute a complex-valed tensor from real-valued tensors denoting its
    amplitude and its phase.

    Dimension
    ---------
    Both inputs must be the same shape (or broadcastable). The output is the
    same shape as the inputs.

    Parameters
    ----------
    ampl : Tensor
        Real-valued array storing complex number amplitudes.
    phase : Tensor
        Real-valued array storing complex number phases in radians.

    Returns
    -------
    complex : Tensor
        Complex numbers formed from the specified amplitudes and phases.
    """
    # TODO : update to use the complex exponential when torch enables it...
    # see here : https://discuss.pytorch.org/t/complex-functions-exp-does- ...
    # not-support-automatic-differentiation-for-outputs-with-complex-dtype/98039
    # Supposedly it was updated, but it still isn't working after calling
    # pip install torch --upgrade
    # https://github.com/pytorch/pytorch/issues/43349
    # https://github.com/pytorch/pytorch/pull/47194
    return ampl * (torch.cos(phase) + 1j * torch.sin(phase))
    #return ampl * torch.exp(phase * 1j)
