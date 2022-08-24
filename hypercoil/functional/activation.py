# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Additional activation functions for neural network layers.
"""
import jax
import jax.numpy as jnp
from typing import Literal, Optional, Tuple, Union
from .utils import (
    complex_decompose, complex_recompose, vmap_over_outer, Tensor
)


def laplace(
    input: Tensor,
    loc: Union[float, Tensor] = 0,
    width: Union[float, Tensor] = 1
) -> Tensor:
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
    return jnp.exp(-jnp.abs(input - loc) / width)


def expbarrier(
    input: Tensor,
    barrier: Union[float, Tensor] = 1
) -> Tensor:
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
    ampl = jnp.abs(input)
    return barrier * jnp.sqrt(1 - jnp.exp(-ampl / barrier ** 2))


# It's not even continuous, let alone differentiable. Let's not use this.
def threshold(
    input: Tensor,
    threshold : Union[Tensor, float],
    dead: Union[Tensor, int] = 0,
    leak: float = 0
) -> Tensor:
    if leak == 0:
        return jnp.where(input > threshold, input, dead)
    return jnp.where(input > threshold, input, dead + leak * input)


def amplitude_laplace(
    input: Tensor,
    loc: Union[float, Tensor] = 0,
    width: Union[float, Tensor] = 1
) -> Tensor:
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
    ampl = jnp.abs(input - loc)
    fact = jnp.exp(-ampl / width) / ampl
    return input * fact


def amplitude_expbarrier(
    input: Tensor,
    barrier: Union[float, Tensor] = 1
) -> Tensor:
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
    barrier : float or broadcastable Tensor (default 0)
        Barrier parameter b of the exponential function.

    Returns
    -------
    out : Tensor
        Transformed input tensor.
    """
    ampl, phase = complex_decompose(input)
    xfm = barrier * jnp.sqrt(1 - jnp.exp(-ampl / barrier ** 2))
    return xfm * jnp.exp(phase * 1j)


def amplitude_tanh(input: Tensor) -> Tensor:
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
    return complex_recompose(jnp.tanh(ampl), phase)


def amplitude_atanh(input: Tensor) -> Tensor:
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
    return complex_recompose(jnp.arctanh(ampl), phase)


def _corrnorm_factor(input: Tensor) -> Tensor:
    factor = jnp.diagonal(input, axis1=-2, axis2=-1)
    factor = (-jnp.sign(factor) * jnp.sqrt(jnp.abs(factor)))
    factor = vmap_over_outer(jnp.outer, 1)((factor, factor))
    return factor + jnp.finfo(input.dtype).eps


def corrnorm(
    input: Tensor,
    factor: Optional[Union[Tensor, Tuple[Tensor, Tensor]]] = None,
    gradpath: Literal['input', 'both'] = 'both'
) -> Tensor:
    r"""
    Correlation normalisation activation function.

    Divide each entry :math:`A_{ij}` of the input matrix :math:`A` by the
    product of the signed square roots of the corresponding diagonals:

    :math:`\bar{A}_{ij} = A_{ij} \frac{\mathrm{sgn}(A_{ii} A_{jj})}{\sqrt{A_{ii}}\sqrt{A_{jj}} + \epsilon}`

    This default behaviour, which maps a covariance matrix to a Pearson
    correlation matrix, can be overriden by providing a ``factor`` argument
    (detailed below). This activation function is also similar to a signed
    version of the normalisation operation for a graph Laplacian matrix.

    :Dimension: **input :** :math:`(*, P, P)`
                    P denotes the row and column dimensions of the input
                    matrices. ``*`` denotes any number of additional
                    dimensions.
                **output :** :math:`(*, P, P)`
                    As above.

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
    if isinstance(factor, jnp.DeviceArray):
        return input / factor
    elif factor is not None:
        factor = jnp.outer(factor[0], factor[1]) + jnp.finfo(input.dtype).eps
    elif gradpath == 'input':
        factor = jax.lax.stop_gradient(_corrnorm_factor(input))
    else:
        factor = _corrnorm_factor(input)
    return input / factor


def isochor(
    input: Tensor,
    volume: float = 1,
    max_condition: Optional[float] = None,
    softmax_temp: Optional[float] = None
) -> Tensor:
    r"""
    Volume-normalising activation function for symmetric, positive definite
    matrices.

    This activation function first finds the eigendecomposition of each input
    matrix. The eigenvalues are then each divided by
    :math:`\sqrt[n]{\frac{v_{in}}{v_{target}}}`
    to normalise the determinant to
    :math:`v_{target}`. Before normalisation, there are options to rescale the
    eigenvalues through a softmax and/or to enforce a maximal condition number
    for the output tensor. If the input tensors are being used to represent
    ellipsoids, for instance, this can constrain the eccentricity of those
    ellipsoids. Finally, the matrix is reconstituted using the original
    eigenvectors and the rescaled eigenvalues.

    :Dimension: **input :** :math:`(*, P, P)`
                    P denotes the row and column dimensions of the input
                    matrices. ``*`` denotes any number of additional
                    dimensions.
                **output :** :math:`(*, P, P)`
                    As above.

    Parameters
    ----------
    input : tensor
        Tensor containing symmetric, positive definite matrices.
    volume : float (default 1)
        Target volume for the normalisation procedure. All output tensors will
        have this determinant.
    max_condition : float :math:`\in [1, \infty)` or None (default None)
        Maximum permissible condition number among output tensors. This can be
        used to constrain the eccentricity of isochoric ellipsoids. To enforce
        this maximum, the eigenvalues of the input tensors are replaced with a
        convex combination of the original eigenvalues and a vector of ones
        such that the largest eigenvalue is no more than ``max_condition``
        times the smallest eigenvalue. Note that a ``max_condition`` of 1
        will always return (a potentially isotropically scaled) identity.
    softmax_temp : float or None (default None)
        If this is provided, then the eigenvalues of the input tensor are
        passed through a softmax with the specified temperature before any
        other processing.

    Returns
    -------
    tensor
        Volume-normalised tensor.
    """
    L, Q = jnp.linalg.eigh(input)
    if softmax_temp is not None:
        L = jax.nn.softmax(L / softmax_temp, axis=-1)
    if max_condition is not None:
        large = L.max(-1, keepdims=True)
        small = L.min(-1, keepdims=True)
        psi = (
            (large - small / max_condition) /
            ((large - 1) - (small - 1) / max_condition)
        )
        psi = jax.nn.relu(psi)
        L = (1 - psi) * L + psi * jnp.ones_like(L)
    Lnorm = jnp.exp((
        jnp.log(L).sum(-1, keepdims=True) - jnp.log(volume)
    ) * (1 / L.shape[-1]))
    L = L / Lnorm
    return Q @ (L[..., None] * Q.swapaxes(-1, -2))
