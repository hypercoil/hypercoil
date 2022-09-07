# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
A hideous, disorganised group of utility functions. Hopefully someday they
can disappear altogether or be moved elsewhere, but for now they exist, a sad
blemish.
"""
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from typing import Callable, Optional, Sequence, Tuple, Union
from ..engine.paramutil import Tensor


#TODO: This will not work if JAX ever adds sparse formats other than BCOO.
def is_sparse(X):
    return isinstance(X, BCOO)


def _conform_vector_weight(weight: Tensor) -> Tensor:
    if weight.ndim == 1:
        return weight
    if weight.shape[-2] != 1:
        return weight[..., None, :]
    return weight


def conform_mask(
    tensor: Tensor,
    mask: Tensor,
    axis: Sequence[int],
    batch=False
) -> Tensor:
    """
    Conform a mask or weight for elementwise applying to a tensor.

    There is almost certainly a better way to do this.

    See also
    --------
    :func:`apply_mask`
    """
    #TODO: require axis to be ordered as in `orient_and_conform`.
    # Ideally, we should create a common underlying function for
    # the shared parts of both operations (i.e., identifying
    # aligning vs. expanding axes).
    tensor = jnp.asarray(tensor)
    if batch and tensor.ndim == 1:
        batch = False
    if isinstance(axis, int):
        if not batch:
            shape_pfx = tensor.shape[:axis]
            mask = jnp.tile(mask, (*shape_pfx, 1))
            return mask
        axis = (axis,)
    if batch:
        axis = (0, *axis)
    # TODO: this feels like it will produce unexpected behaviour.
    mask = mask.squeeze()
    tile = list(tensor.shape)
    shape = [1 for _ in range(tensor.ndim)]
    for i, ax in enumerate(axis):
        tile[ax] = 1
        shape[ax] = mask.shape[i]
    mask = jnp.tile(mask.reshape(*shape), tile)
    return mask


def apply_mask(
    tensor: Tensor,
    msk: Tensor,
    axis: int,
) -> Tensor:
    """
    Mask a tensor along an axis.

    .. warning::

        This function will only work if the mask is one-dimensional. For
        multi-dimensional masks, use :func:`conform_mask`.

    .. warning::

        Use of this function is strongly discouraged. It is incompatible with
        `jax.jit`.

    See also
    --------
    :func:`conform_mask`
    :func:`mask_tensor`
    """
    tensor = jnp.asarray(tensor)
    shape_pfx = tensor.shape[:axis]
    if axis == -1:
        shape_sfx = ()
    else:
        shape_sfx = tensor.shape[(axis + 1):]
    msk = jnp.tile(msk, (*shape_pfx, 1))
    return tensor[msk].reshape(*shape_pfx, -1, *shape_sfx)


def mask_tensor(
    tensor: Tensor,
    mask: Tensor,
    axis: Sequence[int],
    fill_value: Union[float, Tensor] = 0
):
    mask = conform_mask(tensor=tensor, mask=mask, axis=axis)
    return jnp.where(mask, tensor, fill_value)


def wmean(
    input: Tensor,
    weight: Tensor,
    axis: Optional[Union[Sequence[int], int]] = None,
    keepdims: bool = False
) -> Tensor:
    """
    Reducing function for reducing losses: weighted mean.

    >>> wmean(jnp.array([1, 2, 3]), jnp.array([1, 0, 1]))
    DeviceArray(2., dtype=float32)

    >>> wmean(
    ...     jnp.array([[1, 2, 3],
    ...                [1, 2, 3],
    ...                [1, 2, 3]]),
    ...     jnp.array([1, 0, 1]),
    ...     axis=0
    ... )
    DeviceArray([1., 2., 3.], dtype=float32)

    >>> wmean(
    ...     jnp.array([[1, 2, 3],
    ...                [1, 2, 3],
    ...                [1, 2, 3]]),
    ...     jnp.array([1, 0, 1]),
    ...     axis=1,
    ...     keepdims=True
    ... )
    DeviceArray([[2.],
                 [2.],
                 [2.]], dtype=float32)
    """
    if axis is None:
        axis = tuple(range(input.ndim))
    elif isinstance(axis, int):
        axis = (axis,)
    assert weight.ndim == len(axis), (
        'Weight must have as many dimensions as are being reduced')
    retain = [(i not in axis) for i in range(input.ndim)]
    for i, d in enumerate(retain):
        if d: weight = jnp.expand_dims(weight, i)
    wtd = (weight * input)
    return wtd.sum(axis, keepdims=keepdims) / weight.sum(axis, keepdims=keepdims)


#TODO: marking this as an experimental function
def selfwmean(input, dim=None, keepdim=False, gradpath='input', softmax=True):
    """
    Self-weighted mean reducing function. Completely untested. Will break and
    probably kill you in the process.
    """
    if softmax:
        w = jax.nn.softmax(w)
    # I don't think this actually does what we want it to, but this function
    # is actually unsupported, so we won't worry about it yet.
    if gradpath == 'input':
        w = jax.lax.stop_gradient(w)
    elif gradpath == 'weight':
        i = jax.lax.stop_gradient(i)
    return wmean(input=i, weight=w, keepdim=keepdim, gradpath=gradpath)


def complex_decompose(complex: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Decompose a complex-valued tensor into amplitude and phase components.

    :Dimension:
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

    See also
    --------
    :func:`complex_recompose`
    """
    ampl = jnp.abs(complex)
    phase = jnp.angle(complex)
    return ampl, phase


def complex_recompose(ampl: Tensor, phase: Tensor) -> Tensor:
    """
    Reconstitute a complex-valued tensor from real-valued tensors denoting its
    amplitude and its phase.

    :Dimension:
        Both inputs must be the same shape (or broadcastable). The
        output is the same shape as the inputs.

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

    See also
    --------
    :func:`complex_decompose`
    """
    # TODO : consider using the complex exponential function,
    # depending on the gradient properties
    #return ampl * jnp.exp(phase * 1j)
    return ampl * (jnp.cos(phase) + 1j * jnp.sin(phase))


def amplitude_apply(func: Callable) -> Callable:
    """
    Decorator for applying a function to the amplitude component of a complex
    tensor.
    """
    def wrapper(complex: Tensor) -> Tensor:
        ampl, phase = complex_decompose(complex)
        return complex_recompose(func(ampl), phase)
    return wrapper


def orient_and_conform(
    input: Tensor,
    axis: Union[int, Sequence[int]],
    reference: Optional[Tensor] = None,
    dim: Optional[int] = None
) -> Tensor:
    """
    Orient an input tensor along a set of axes, and conform its overall
    dimension to equal that of a reference.

    .. warning::

        If both ``reference`` and ``dim`` are provided, then ``dim`` takes
        precedence.

    Parameters
    ----------
    input : tensor
        Input tensor.
    axis : tuple
        Output axes along which the tensor's input dimensions should be
        reoriented. This should be an n-tuple, where n is the number of axes
        in the input tensor. These axes must be in the same order in the input
        tensor; if they are not, the input must be transposed before being
        oriented.
    reference : tensor or None
        Reference tensor. The output is unsqueezed so that its total
        dimension equals that of the reference. Either a reference or an
        explicit output dimension (``dim``) must be provided.
    dim : int or None
        Number of tensor axes in the desired output.

    Returns
    -------
    tensor
        Reoriented tensor with singleton axes appended to conform with the
        reference number of axes.
    """
    if isinstance(axis, int):
        axis = (axis,)
    if dim is None and reference is None:
        raise ValueError('Must specify either `reference` or `dim`')
    elif dim is None:
        dim = reference.ndim
    # can't rely on this when we compile with jit
    assert len(axis) == input.ndim, (
        'Output orientation axis required for each input dimension')
    shape = [1] * dim
    asgn = [0] * dim
    for size, ax in zip(input.shape, axis):
        shape[ax] = size
        assert sum(asgn[ax:]) == 0, (
            'All axes must be in order. Transpose the input if necessary.')
        asgn[ax] = 1
    return input.reshape(*shape)
