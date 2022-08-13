# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
A hideous, disorganised group of utility functions. Hopefully someday they
can disappear altogether or be moved elsewhere, but for now they exist, a sad
blemish.
"""
import jax.numpy as jnp
import torch
from jax import vmap
from jax.tree_util import tree_map, tree_reduce
from jax.experimental.sparse import BCOO
from functools import partial, reduce
from typing import Any, Callable, Optional, Sequence, Tuple, Union


#TODO: replace with jaxtyping at some point
Tensor = Any
PyTree = Any


def atleast_4d(*pparams) -> Tensor:
    res = []
    for p in pparams:
        if p.ndim == 0:
            result = p.reshape(1, 1, 1, 1)
        elif p.ndim == 1:
            result = p[None, None, None, ...]
        elif p.ndim == 2:
            result = p[None, None, ...]
        elif p.ndim == 3:
            result = p[None, ...]
        else:
            result = p
        res.append(result)
        if len(res) == 1:
            return res[0]
    return res


#TODO: This will not work if JAX ever adds sparse formats other than BCOO.
def is_sparse(X):
    return isinstance(X, BCOO)


def _conform_vector_weight(weight: Tensor) -> Tensor:
    if weight.ndim == 1:
        return weight
    if weight.shape[-2] != 1:
        return weight[..., None, :]
    return weight


def _dim_or_none(x, i):
    proposal = i + x
    if proposal < 0:
        return None
    return proposal


def apply_vmap_over_outer(
    x: PyTree,
    f: Callable,
    f_dim: int
) -> Tensor:
    """
    Apply a tensor-valued function to the outer dimensions of a tensor.
    """
    ndim = tree_map(lambda x: x.ndim - f_dim - 1, x)
    ndmax = tree_reduce(max, ndim)
    #print([(
    #    tree_map(partial(_dim_or_none, i=i - ndmax), ndim), i)
    #    for i in range(0, ndmax + 1)
    #])
    return reduce(
        lambda x, g: g(x),
        [partial(
            vmap,
            in_axes=tree_map(partial(_dim_or_none, i=i - ndmax), ndim),
            out_axes=i
        ) for i in range(0, ndmax + 1)],
        f
    )(*x)


def vmap_over_outer(f: Callable, f_dim: int) -> Callable:
    """
    Transform a function to apply to the outer dimensions of a tensor.
    """
    return partial(apply_vmap_over_outer, f=f, f_dim=f_dim)


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
    i = input.clone()
    w = input.clone()
    if softmax:
        w = torch.softmax(w)
    if gradpath == 'input':
        w = w.detach()
    elif gradpath == 'weight':
        i = i.detach()
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


def _promote_nnz_dim(values):
    return values.permute(list(range(values.dim()))[::-1])
    # slightly faster but not as easy to implement
    #return values.permute((*list(range(values.dim()))[1:], 0))


def _demote_nnz_dim(values):
    return _promote_nnz_dim(values)
    #return values.permute((-1, *list(range(values.dim()))[:-1]))


def _conform_dims(A_values, B_values):
    A_shape = A_values.shape[1:]
    B_shape = B_values.shape[1:]
    missing_dims = len(A_shape) - len(B_shape)
    if missing_dims == 0:
        return A_values, B_values
    elif missing_dims > 0:
        B_shape = [1] * missing_dims + list(B_shape)
        return A_values, B_values.view(-1, *B_shape)
    else:
        A_shape = [1] * -missing_dims + list(A_shape)
        return A_values.view(-1, *A_shape), B_values


#TODO: marking this as an experimental function
def sparse_mm(A, B):
    """
    Batched sparse-sparse matrix multiplication.

    .. admonition:: Dimensions and broadcasting

        The dense dimension of each input COO tensor (that is, the dimension
        of the tensor in ``tensor._values()``) is equal to ``(nnz, *)``, where
        ``nnz`` denotes the number of nonzero elements in the tensor. For the
        matrix multiply, array broadcasting can be performed using the
        remaining dense dimensions after the ``nnz`` dimension. For instance,
        an array ``A`` with dense size (20, 3, 5) can be broadcast with an
        array ``B`` that has dense size (35, 5) or (17, 1, 1, 5), but not one
        that has dense size (20, 3). (The first dense axis size, ``nnz``, is
        ignored when broadcasting.)

    :Dimension: **A :** :math:`(M, K, *)`
                    `*` denotes any number of trailing (dense) dimensions,
                    potentially including a batch dimension. Note that the
                    batch dimension must be a dense dimension. M and K denote
                    the sparse dimensions of the input tensor, corresponding
                    to the axes of the matrix to be multiplied. Note that the
                    sparse dimension must be exactly 2.
                **B :** :math:`(K, N, *)`
                    N denotes the outer sparse dimension of the second matrix
                    batch.
                **Output :** :math:`(M, N, *)`
                    As above.

    Parameters
    ----------
    A, B : sparse COO tensor
        Tensors to be multiplied.

    Returns
    -------
    sparse COO tensor
        Sparse product of sparse tensors.

    See: https://github.com/rusty1s/pytorch_sparse/issues/147
    """
    m = A.shape[0]
    n = B.shape[1]
    k = A.shape[1]
    assert B.shape[0] == k, (
        f'Inner matrix dimensions {A.shape[1]} and {B.shape[1]} '
        'must agree')
    #assert A.dense_dim() == B.dense_dim()
    A = A.coalesce()
    B = B.coalesce()
    A_values = A.values()
    B_values = B.values()
    A_indices = A.indices()
    B_indices = B.indices()
    A_values, B_values = _conform_dims(A_values, B_values)
    A_values = _demote_nnz_dim(A_values)
    B_values = _demote_nnz_dim(B_values)
    out_indices, out_values = _sparse_mm(
        A_indices, A_values, B_indices, B_values, m, k, n)
    out_values = _promote_nnz_dim(out_values)
    o = out_values.shape[1:]
    return torch.sparse_coo_tensor(
        indices=out_indices, values=out_values, size=(m, n, *o)
    )


def _sparse_mm(A_indices, A_values, B_indices, B_values, m, k, n):
    if A_values.dim() <= 1:
        out = torch.sparse.mm(
            torch.sparse_coo_tensor(
                indices=A_indices, values=A_values, size=(m, k)),
            torch.sparse_coo_tensor(
                indices=B_indices, values=B_values, size=(k, n)),
        ).coalesce()
        return out.indices(), out.values()
    else:
        if len(A_values) != len(B_values):
            if len(A_values) == 1:
                A_values = [A_values[0]] * len(B_values)
            elif len(B_values) == 1:
                B_values = [B_values[0]] * len(A_values)
            else:
                raise RuntimeError(
                    'Dense dimensions of arrays are incompatible: '
                    f'{A_values.shape} and {B_values.shape}')
        out = [
            _sparse_mm(A_indices, a, B_indices, b, m, k, n)
            for a, b in zip(A_values, B_values)
        ]
        out_indices, out_values = zip(*out)
        return out_indices[0], torch.stack(out_values)


#TODO: marking this as an experimental function
def sparse_rcmul(A, R=None, C=None, coalesce_output=True):
    """
    Batchable row- and column-wise multiplication of sparse matrices.

    .. note::
        Regardless of their form (sparse or dense) at call time, inputs ``R``
        and ``C`` will be cast to dense by this operation. Either ``R`` or
        ``C`` or both must be provided as inputs.

    Parameters
    ----------
    A : sparse COO tensor
        Matrix or matrix batch to be multiplied. ``A`` must have exactly 2
        sparse dimensions and can have any number of dense dimensions.
    R : tensor (default None)
        Row-wise multiplier. All elements in row i of input ``A`` are
        multiplied by the corresponding entry i of ``R``.
    C : tensor (default None)
        Column-wise multiplier. All elements in column j of input ``A`` are
        multiplied by the corresponding entry j of ``C``.
    coalesce_output : bool (default True)
        Indicates that the output should be coalesced before it is returned.

    Returns
    -------
    sparse COO tensor
        Product.
    """
    if R is not None and C is not None:
        R = _rcmul_broadcast(R, A._indices()[0])
        C = _rcmul_broadcast(C, A._indices()[1])
        RC = R * C
    elif R is not None:
        RC = _rcmul_broadcast(R, A._indices()[0])
    elif C is not None:
        RC = _rcmul_broadcast(C, A._indices()[1])
    else:
        raise ValueError(
            'Must specify either row or column multiplier or both')
    out_values = A._values() * RC
    out = torch.sparse_coo_tensor(
        indices=A._indices(),
        values=out_values,
        size=A.size()
    )
    if coalesce_output:
        return out.coalesce()
    return out


def _rcmul_broadcast(tensor, indices):
    if tensor.is_sparse:
        tensor = tensor.to_dense()
    return tensor[indices]


#TODO: marking this as an experimental function
def sparse_reciprocal(A):
    """
    Reciprocal of nonzero elements in a sparse tensor. Zero-valued elements
    are mapped back to zero.
    """
    if not A.is_sparse:
        out =  A.reciprocal()
        out[A == 0] = 0
        return out
    coalesce_output = A.is_coalesced()
    values = A._values().reciprocal()
    values[A._values() == 0] = 0
    out = torch.sparse_coo_tensor(
        indices=A._indices(),
        values=values,
        size=(A.size())
    )
    if coalesce_output:
        return out.coalesce()
    return out


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
