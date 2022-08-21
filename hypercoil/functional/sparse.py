# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities for operations on various BCOO sparse matrix formats.

Top-k BCOO format
-----------------

One common BCOO sparse format that is useful in many applications, such as
connectopic mapping, is the top-k format. Sparse matrices in top-k format have
the following properties:

- Each row has no more than k non-zero entries.
- The indices of nonzero entries are shared across all batch elements.
- The indexing tensor has shape (``...``, ``n_rows``, ``k``, 1) where ``...``
  indicates a number of leading singleton dimensions equal to the number of
  ``channel_dims`` + 1. (``n_rows`` can be substituted for a singleton
  dimension as well, in which case all rows have the same nonzero indices.)
- The data tensor has shape
  (``batch_size``, ``*channel_dims``, ``n_rows``, ``k``).
"""
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.tree_util import tree_map
from jax.experimental.sparse import BCOO
from typing import Any, Callable, Literal, Optional, Sequence, Tuple, Union

from .utils import (
    Tensor, standard_axis_number, vmap_over_outer,
    fold_and_promote, demote_and_unfold
)


TopKTensor = Any


def _ix(x, i): return x[i]


def random_sparse(
    shape: Sequence[int],
    k: Optional[int] = None,
    *,
    key: jax.random.PRNGKey,
) -> Tensor:
    """
    Generate a batch of random sparse matrices in the top-k format.
    """
    ikey, dkey = jax.random.split(key)
    if k is None: k = max(shape[-1] // 100, 1)
    if len(shape) > 2:
        batch_size, *channel_dims, n_rows, n_cols = shape
        idx_unsqueeze = tuple([None] * (1 + len(channel_dims)) + [...])
        data = jax.random.normal(dkey, (batch_size, *channel_dims, n_rows, k))
    else:
        n_rows, n_cols = shape
        idx_unsqueeze = (...)
        data = jax.random.normal(dkey, (n_rows, k))
    ikeys = jax.random.split(ikey, n_rows)
    indices = jnp.stack([
        jax.random.choice(i, a=n_cols, shape=(k, 1), replace=False)
        for i in ikeys
    ], axis=0)
    print(indices.shape, idx_unsqueeze)
    return BCOO((data, indices[idx_unsqueeze]), shape=shape).sum_duplicates()


def sparse_astype(
    tensor: Tensor,
    dtype: Any
) -> Tensor:
    """
    Set the data type of a sparse matrix.

    This function is probably unnecessary, but I'm missing a way to do this
    with the current JAX API.
    """
    if tensor.dtype == dtype:
        return tensor
    return BCOO(
        (tensor.data.astype(dtype), tensor.indices),
        shape=tensor.shape
    )


def spdiagmm(
    lhs: Union[Tensor, TopKTensor],
    rhs: Union[Tensor, TopKTensor],
    lhs_diag: bool = False
) -> TopKTensor:
    """
    Matrix multiplication of a top-k format sparse matrix with a diagonal
    matrix. Returns a sparse matrix in the top-k format.

    The diagonal matrix should be formatted as a vector or batch of vectors
    whose final dimension is equal to the matching inner dimension of the
    top-k sparse matrix. All other dimensions should be broadcastable.

    A diagonal matrix that is in matrix form can be converted into a vector by
    calling a ``diagonal`` function that is appropriately vmapped.
    """
    if lhs_diag:
        data = lhs[..., None] * rhs.data
        indices = rhs.indices
        shape = rhs.shape
    else:
        data = lhs.data * rhs[..., tuple(lhs.indices.squeeze())]
        indices = lhs.indices
        shape = lhs.shape
    return BCOO((data, indices), shape=shape)


def spspmm(
    lhs: TopKTensor,
    rhs: TopKTensor,
    indices: Optional[Tensor] = None,
    n_blocks: int = 1
):
    """
    Sparse-sparse matrix multiplication of top-k format sparse matrices.

    This function is a wrapper around the JAX sparse matrix multiplication
    function ``bcoo_dot_general``. It is a convenience function that also
    provides the option to separate the matrix multiplication into blocks for
    serialisation and reduced memory usage. It additionally provides the
    option to use a pre-computed index tensor and return a top-k format sparse
    matrix containing the results of the matrix multiplication at only the
    specified indices.

    .. warning::
        Note that this implementation of the matrix multiplication operation
        returns
        :math:`A B^\intercal` for LHS :math:`A` and RHS :math:`B`.

    Parameters
    ----------
    lhs : TopKTensor
        Left-hand side sparse matrix in the top-k format.
    rhs : TopKTensor
        Right-hand side sparse matrix in the top-k format.
    indices : Tensor or None (default: None)
        Indices of the matrix product to return. If not specified, the entire
        matrix is returned.
    n_blocks : int (default: 1)
        Number of blocks to split the matrix multiplication into for
        serialisation. If set to 1, the matrix multiplication is performed
        directly.

    Returns
    -------
    TopKTensor or Tensor
        Result of the matrix multiplication.
    """
    if indices is None:
        return spspmm_full(lhs, rhs)
    elif n_blocks == 1:
        return topkx(spspmm_full, auto_index=False)(indices, lhs, rhs)
    else:
        return _serialised_spspmm(
            lhs=lhs,
            rhs=rhs,
            indices=indices,
            n_blocks=n_blocks
        )


def spspmm_full(
    lhs: TopKTensor,
    rhs: TopKTensor
) -> Tensor:
    """
    Matrix multiplication of a top-k format sparse matrix with another sparse
    matrix in the top-k format. Returns a full matrix.

    .. warning::
        Note that this implementation of the matrix multiplication operation
        returns
        :math:`A B^\intercal` for LHS :math:`A` and RHS :math:`B`.
    """
    contracting_dims = ((lhs.ndim - 1,), (rhs.ndim - 1,))
    batch_dims = (tuple(range(lhs.ndim - 2)), tuple(range(rhs.ndim - 2)))
    return jax.experimental.sparse.bcoo_dot_general(
        lhs, rhs, dimension_numbers=(contracting_dims, batch_dims)
    ).data.squeeze(-1)


def select_indices(
    tensor: Tensor,
    threshold: float = 0.0,
    threshold_type: Literal['abs>', 'abs<' '>', '<'] = 'abs>',
    top_k: bool = True,
    top_k_reduction: Optional[Literal['mean']] = 'mean',
    fix_indices_over_channel_dims: bool = True,
) -> Tensor:
    """
    Select indices from a tensor (e.g., in preparation for sparsification).

    The input can be a boolean matrix, in which case the output contains the
    indices of True entries. The input can also be a matrix with values, in
    which case the output contains the indices of entries that survive the
    thresholding operation.

    .. warning::
        This function is not compatible with JIT compilation.

    .. warning::
        If the input is batched or contains multiple channels, the ``top_k``
        option will return separate indices for each channel and each batch
        element. Ensure that ``top_k_reduction`` is set to ``'mean'`` to
        obtain a single index across all batch elements (and potentially
        channels, according to ``fix_indices_over_channel_dims``).

    Parameters
    ----------
    tensor : Tensor
        The input tensor.
    threshold : float (default: 0.0)
        The threshold value. Used only if the input matrices are matrices with
        values.
    threshold_type : one of 'abs>', 'abs<', '>', '<' (default: 'abs>')
        The type of thresholding operation to perform.
    top_k : bool (default: False)
        If True, then the threshold value must be an integer, and the
        thresholding operation will be replaced by selection of the top k
        entries.
    fix_indices_over_channel_dims : bool (default: True)
        If True, then the indices of nonzero entries that are returned will
        be fixed over all channel dimensions. If False, then the indices of
        nonzero entries that are returned are allowed to vary over all channel
        dimensions.
    """
    if fix_indices_over_channel_dims:
        fixed_axes = tuple(range(tensor.ndim - 2))
    else:
        fixed_axes = (0,)
    if tensor.dtype == jnp.bool_:
        data = tensor.any(axis=fixed_axes)
        return jnp.stack(jnp.where(tensor), axis=-1)
    elif not top_k:
        if threshold_type == 'abs>':
            tensor = jnp.abs(tensor) > threshold
        elif threshold_type == 'abs<':
            tensor = jnp.abs(tensor) < threshold
        elif threshold_type == '>':
            tensor = tensor > threshold
        elif threshold_type == '<':
            tensor = tensor < threshold
        tensor = tensor.any(axis=fixed_axes)
        return jnp.stack(jnp.where(tensor), axis=-1)
    else:
        if top_k_reduction == 'mean':
            tensor = tensor.mean(axis=fixed_axes)
        if not isinstance(threshold, int):
            raise ValueError(
                'If topk is True, then the threshold value must be an integer.'
            )
        if threshold_type == 'abs>':
            descending = True
            tensor = jnp.abs(tensor)
        elif threshold_type == 'abs<':
            descending = False
            tensor = jnp.abs(tensor)
        elif threshold_type == '>':
            descending = True
        elif threshold_type == '<':
            descending = False
        return topk(tensor, k=threshold, axis=-1, descending=descending)[..., None]


def trace_spspmm(
    lhs: TopKTensor,
    rhs: TopKTensor,
    threshold: float = 0.0,
    threshold_type: Literal['abs>', 'abs<' '>', '<'] = 'abs>',
    top_k: bool = True,
    top_k_reduction: Optional[Literal['mean']] = 'mean',
    fix_indices_over_channel_dims: bool = True,
) -> Tensor:
    """
    Trace the matrix multiplication of two top-k format sparse matrices to
    determine the indices of nonzero entries.

    The inputs can be boolean matrices, in which case the output contains the
    indices of True entries. The inputs can also be matrices with values, in
    which case the output contains the indices of entries that survive the
    thresholding operation.

    .. warning::
        This function is not compatible with JIT compilation.

    .. warning::
        If the input is batched or contains multiple channels, the ``top_k``
        option will return separate indices for each channel and each batch
        element. Ensure that ``top_k_reduction`` is set to ``'mean'`` to
        obtain a single index across all batch elements (and potentially
        channels, according to ``fix_indices_over_channel_dims``).

    Parameters
    ----------
    lhs : TopKTensor
        The left-hand side sparse matrix.
    rhs : TopKTensor
        The right-hand side sparse matrix.
    threshold : float (default: 0.0)
        The threshold value. Used only if the input matrices are matrices with
        values.
    threshold_type : one of 'abs>', 'abs<', '>', '<' (default: 'abs>')
        The type of thresholding operation to perform.
    top_k : bool (default: False)
        If True, then the threshold value must be an integer, and the
        thresholding operation will be replaced by selection of the top k
        entries.
    fix_indices_over_channel_dims : bool (default: True)
        If True, then the indices of nonzero entries that are returned will
        be fixed over all channel dimensions. If False, then the indices of
        nonzero entries that are returned are allowed to vary over all channel
        dimensions.
    """
    data = spspmm_full(lhs, rhs)
    return select_indices(
        data,
        threshold=threshold,
        threshold_type=threshold_type,
        top_k=top_k,
        top_k_reduction=top_k_reduction,
        fix_indices_over_channel_dims=fix_indices_over_channel_dims,
    )


def _null_carrier(carry, pparams, retvals):
    """
    No-op carrier function for block serialisation transformations of
    functions.
    """
    return carry


def _noop(retvals):
    """
    No-op postprocessing function for block serialisation transformations of
    functions.
    """
    return retvals


def block_serialise(
    f: Callable,
    *,
    n_blocks: int = 1,
    argnums: Sequence[int] = (0,),
    retnums: Sequence[int] = (0,),
    in_axes: Sequence[int] = (-1,),
    out_axes: Sequence[int] = (-1,),
    carrier_fn: Optional[Callable] = None,
    carry_init: Any = None,
    return_carry: bool = False,
    postprocess_fn: Optional[Callable] = None,
) -> Callable:
    """
    Serialise a function to be run over blocks of data, in order to reduce the
    memory footprint of each call.

    .. warning::
        Each specified input argument must be divisible by the number of
        blocks along the specified axis.

    .. warning::
        Any parameters that are not to be serialised must be passed as keyword
        arguments. If this is not possible using the original function, then
        you will have to write a wrapper function.
    """
    if len(in_axes) == 1:
        in_axes = in_axes * len(argnums)
    if len(out_axes) == 1:
        out_axes = out_axes * len(retnums)
    if carrier_fn is None:
        carrier_fn = _null_carrier
    if postprocess_fn is None:
        postprocess_fn = _noop

    def _f_scan_compat(carry, blocked_pparams, **params):
        retvals = f(*blocked_pparams, **params)
        carry = carrier_fn(carry, blocked_pparams, retvals)
        return carry, postprocess_fn(retvals)

    def _f_serialised(*pparams, **params):
        pparams = list(pparams)
        for arg, ax in zip(argnums, in_axes):
            pparams[arg] = fold_and_promote(
                pparams[arg],
                axis=ax,
                n_folds=n_blocks)
        carry, out = jax.lax.scan(
            partial(_f_scan_compat, **params),
            carry_init,
            pparams,
        )
        if not isinstance(out, tuple):
            out = (out,)
        out = list(out)
        for ret, ax in zip(retnums, out_axes):
            ax = standard_axis_number(ax, out[ret].ndim)
            out[ret] = demote_and_unfold(out[ret], ax, (ax - 1, ax))
        if len(out) == 1:
            if return_carry:
                return carry, out[0]
            return out[0]
        if return_carry:
            return carry, tuple(out)
        return tuple(out)

    return _f_serialised


# Block serialisation for top-k format sparse matrices.
# ----------------------------------------------------------------------------
# This is fairly complicated because it involves packaging and unpacking the
# data and indices into BCOO tensors, as well as tracking the shape of the
# output.
# Update: we've given up on tracking the shape of the output, and instead
# require the caller to pass the shape of the output.

# So it turns out that carrying the shape of the output is not possible with
# JIT compilation. Shapes must be static, but accumulating the shape of the
# output through a lax scan wraps it into a DeviceArray. So keeping it static
# is not possible. This means that we must require the caller to pass the
# shape of the output.
# def _shape_carrier(carry, _, retvals, retnums=(0,), axes=(-1,)):
#     """
#     Carrier function for the shape of the output. This can be used to track
#     the shape of the output of a block serialised function when the output is
#     a sparse matrix.
#     """
#     axes = tuple([
#         standard_axis_number(ax, retvals[i].ndim)
#         for ax, i in zip(axes, retnums)
#     ])
#     axidx = tuple([
#         jnp.where(jnp.arange(retvals[i].ndim) == ax, True, False)
#         for ax, i in zip(axes, retnums)])
#     update = tuple([jnp.array(retvals[i].shape) for i in retnums])
#     return tree_map(
#         lambda ax, cur, new: jnp.where(ax, cur + new, new),
#         axidx, carry, update)


def _shape_block(shape, n_blocks, axis=-2):
    return shape[:axis] + (shape[axis] // n_blocks,) + shape[axis + 1:]


def sp_block_serialise(
    f: Callable,
    *,
    n_blocks: int = 1,
    argnums: Sequence[int] = (),
    retnums: Sequence[int] = (),
    in_axes: Sequence[int] = (),
    out_axes: Sequence[int] = (),
    sp_argnums: Sequence[int] = (0,),
    sp_retnums: Sequence[int] = (0,),
    sp_retshapes: Sequence[Sequence[int]] = (),
) -> Callable:
    """
    Function block serialisation transformation with a convenience wrapper for
    handling top-k format sparse data.
    """
    def _cfg_return():
        oax = 0
        retnum = 0
        for i in range(len(retnums + sp_retnums)):
            if i in retnums:
                yield retnum, out_axes[oax]
                oax += 1
                retnum += 1
            else:
                yield retnum, -2
                yield retnum + 1, -3
                retnum += 2

    if sp_retshapes == () and sp_retnums != ():
        raise ValueError('Must specify shapes of any sparse outputs')
    retnums, out_axes = zip(*_cfg_return())
    retnums, out_axes = tuple(retnums), tuple(out_axes)
    #print(retnums, out_axes)

    def _prepare_transformation(pparams):
        j = 0
        iax = 0
        for i, pparam in enumerate(pparams):
            if i in sp_argnums:
                blocked_shape = _shape_block(pparam.shape, n_blocks)
                yield pparam.data, j, -2, (True, blocked_shape)
                yield pparam.indices, j + 1, -3, (False, None)
                j += 2
            else:
                yield pparam, j, in_axes[iax], (False, None)
                j += 1
                iax += 1

    def _finalise_transformation(shapes, retvals):
        s = 0
        for i, retval in enumerate(retvals):
            if i in sp_retnums:
                #print(s, shapes[s])
                indices = retvals[i + 1]
                #yield ((retval, indices), shapes[s])
                #yield _mk_bcoo(retvals, indices, shapes[s])
                yield BCOO((retval, indices), shape=shapes[s])
                s += 1
            elif not (i - 1) in sp_retnums:
                yield retval

    def _package_as_topk(pparams, addresses):
        prev = False
        for i, a in enumerate(addresses):
            cur, shape = a
            if cur:
                #yield _mk_bcoo(pparams[i], pparams[i + 1], shape=shape)
                yield BCOO((pparams[i], pparams[i + 1]), shape=shape)
            elif not prev:
                yield pparams[i]
            prev = cur

    def _unpack_topk(retvals):
        for i, retval in enumerate(retvals):
            if i in sp_retnums:
                yield retval.data
                yield retval.indices
            else:
                yield retval

    def _unpack_topk_postprocess(retvals):
        return tuple(_unpack_topk(retvals))

    def _f_unpack(*pparams, addresses, **params):
        pparams = list(_package_as_topk(pparams, addresses))
        out = f(*pparams, **params)
        if not isinstance(out, tuple):
            out = (out,)
        return out

    # def _init_shapes(data):
    #     if sp_retndims is None:
    #         for i, d in data:
    #             if i:
    #                 yield jnp.zeros(len(d))
    #                 #yield tuple([0 for _ in d])
    #     else:
    #         for ndim in sp_retndims:
    #             yield jnp.zeros(ndim)

    def _f_serialised(*pparams, **params):
        inputs = _prepare_transformation(pparams)
        pparams, argnums, in_axes, data_addresses = zip(*inputs)
        #zero_shapes = tuple(_init_shapes(data_addresses))
        #print(zero_shapes)
        serialised = block_serialise(
            partial(_f_unpack, addresses=data_addresses),
            n_blocks=n_blocks,
            argnums=argnums, retnums=retnums,
            in_axes=in_axes, out_axes=out_axes,
            # carrier_fn=partial(
            #     _shape_carrier,
            #     axes=(-2,) * len(sp_retnums),
            #     retnums=sp_retnums),
            # carry_init=zero_shapes,
            # return_carry=True,
            postprocess_fn=_unpack_topk_postprocess,
        )
        #shapes, retvals = serialised(*pparams, **params)
        retvals = serialised(*pparams, **params)
        #print('shapes ', shapes)
        if not isinstance(retvals, tuple):
            retvals = (retvals,)
        #return shapes, retvals
        out = tuple(_finalise_transformation(sp_retshapes, retvals))
        if len(out) == 1:
            return out[0]
        return tuple(out)

    return _f_serialised
# ----------------------------------------------------------------------------


def _serialised_spspmm(
    lhs: TopKTensor,
    rhs: TopKTensor,
    indices: Tensor,
    n_blocks: int = 1,
):
    if lhs.shape[-2] % n_blocks != 0:
        raise ValueError(
            'The number of blocks must divide the number of rows in the '
            'left-hand side matrix.'
        )
    lhs_data = fold_and_promote(lhs.data, axis=-2, n_folds=n_blocks)
    lhs_indices = fold_and_promote(lhs.indices, axis=-3, n_folds=n_blocks)
    out_indices = fold_and_promote(indices, axis=-3, n_folds=n_blocks)
    lhs_shape = (
        lhs.shape[:-2] + (lhs.shape[-2] // n_blocks, ) + lhs.shape[-1:])

    _, out_data = jax.lax.scan(
        partial(_serialised_spspmm_impl, rhs=rhs, lhs_shape=lhs_shape),
        None,
        (lhs_data, lhs_indices, out_indices))
    out_data = demote_and_unfold(out_data, -2, (-3, -2))
    out_shape = lhs.shape[:-2] + (lhs.shape[-2], rhs.shape[-2])
    out_idx_idx = tuple(
        [None] * (out_data.ndim - indices.ndim + 1) + [Ellipsis])
    return BCOO((out_data, indices[out_idx_idx]), shape=out_shape)


def _serialised_spspmm_impl(
    _: None,
    data: Tuple[Tensor, Tensor, Tensor],
    rhs: TopKTensor,
    lhs_shape: Tuple[int, ...]
):
    lhs_data, lhs_indices, out_indices = data
    out = spspmm_full(
        BCOO((lhs_data, lhs_indices), shape=(lhs_shape)), rhs
    )
    sampling_fn = vmap_over_outer(_ix, 1)
    return None, sampling_fn((out, out_indices.squeeze(-1)))


def topkx(
    f: Callable,
    *,
    retnums: Sequence[int] = (0,),
    auto_index: bool = False,
    threshold_type: Literal['abs>', 'abs<' '>', '<'] = 'abs>',
    fix_indices_over_channel_dims: bool = True,
) -> TopKTensor:
    """
    Transform a function that produces a full matrix so that it instead
    produces a sparse matrix in the top-k format.

    .. warning::
        The transformed function produces as an intermediate result a full
        matrix. Thus, this transformation will not reduce the memory usage of
        the function.

    .. warning::
        The transformed function is compatible with JIT compilation only if
        ``auto_index`` is False. Indices can be obtained in a separate step
        using the :func:`select_indices` function.

    .. note::
        The transformed function requires an additional first argument, which
        depends on the value of ``auto_index``. If ``auto_index`` is True, the
        first argument should be the value of k. Otherwise, the first argument
        should be the indices of the top k elements.

    Parameters
    ----------
    f : Callable
        The function to transform.
    retnums : Sequence[int] (default: (0,))
        The indices of the return values to be converted to the top-k format.
    auto_index : bool (default: False)
        If True, the indices of the return values will be automatically
        generated by the transformed function.
    indices : Tensor (default: None)
        The indices of the nonzero entries in the top-k format. If None, then
        the indices are sampled from the output of the function according to
        the thresholding operation.
    threshold_type : one of 'abs>', 'abs<', '>', '<' (default: 'abs>')
        The type of thresholding operation to perform.
    fix_indices_over_channel_dims : bool (default: True)
        If True, then the indices of nonzero entries that are returned will
        be fixed over all channel dimensions. If False, then the indices of
        nonzero entries that are returned are allowed to vary over all channel
        dimensions.

    Returns
    -------
    Callable
        The transformed function. Any tensor return values specified by
        ``retnums`` will be returned in the top-k sparse format when the
        transformed function is called. If ``auto_index`` is True, then the
        transformed function takes an additional first argument, which is the
        value of k to use in the top-k selection. Otherwise, the transformed
        function takes an additional first argument, which is the indices of
        the top k entries to include in the output.
    """
    sampling_fn = vmap_over_outer(_ix, 1)
    if auto_index:
        def _f_and_sample(k, *pparams, **params):
            f_out = f(*pparams, **params)
            if not isinstance(f_out, tuple):
                f_out = (f_out,)
            out_transformed = [out for out in f_out]
            for idx in retnums:
                indices = select_indices(
                    f_out[idx],
                    top_k=True,
                    threshold=k,
                    threshold_type=threshold_type,
                    fix_indices_over_channel_dims=fix_indices_over_channel_dims,
                )
                data = sampling_fn((f_out[idx], indices.squeeze(-1)))
                idx_idx = tuple(
                    [None] * (data.ndim - indices.ndim + 1) + [Ellipsis])
                out_transformed[idx] = BCOO(
                    (data, indices[idx_idx]),
                    shape=f_out[idx].shape
                )
            if len(out_transformed) == 1:
                return out_transformed[0]
            return tuple(out_transformed)
    else:
        def _f_and_sample(indices, *params, **pparams):
            f_out = f(*params, **pparams)
            if not isinstance(f_out, tuple):
                f_out = (f_out,)
            out_transformed = [out for out in f_out]
            for idx in retnums:
                data = sampling_fn((f_out[idx], indices.squeeze(-1)))
                idx_idx = tuple(
                    [None] * (data.ndim - indices.ndim + 1) + [Ellipsis])
                out_transformed[idx] = BCOO(
                    (data, indices[idx_idx]),
                    shape=f_out[idx].shape
                )
            if len(out_transformed) == 1:
                return out_transformed[0]
            return tuple(out_transformed)
    return _f_and_sample


def topk(
    tensor: Tensor,
    k: int,
    *,
    axis: int = -1,
    descending: bool = True,
) -> Tensor:
    """
    Select the top k entries of a tensor and return the indices in a format
    compatible with the top-k sparse matrix format.
    """
    if descending:
        tensor = -tensor
    arr = np.array(tensor)
    slc = [slice(None)] * arr.ndim
    slc[axis] = slice(None, k)
    slc = tuple(slc)
    return np.argpartition(arr, k - 1, axis=axis)[slc]


def as_topk(
    tensor: Tensor,
    k: int,
    *,
    descending: bool = True,
) -> TopKTensor:
    """
    Convert a tensor to a top-k sparse matrix format.
    """
    #TODO: allow user to specify axis (?)
    indices = topk(tensor, k, axis=-1, descending=descending)
    data_fn = vmap_over_outer(_ix, 1)
    data = data_fn((tensor, indices))
    return BCOO((data, indices[..., None]), shape=tensor.shape)


def full_as_topk(
    tensor: Tensor,
) -> TopKTensor:
    """
    Represent a batch of full tensors in top-k sparse matrix format.

    This is strictly less efficient than standard full tensor format, but
    provides compatibility with functions that operate on top-k sparse
    matrices.
    """
    data = tensor
    ndim = data.ndim
    k = data.shape[-1]
    indices = jnp.arange(k)
    idx_idx = tuple([None] * (ndim - 1) + [Ellipsis, None])
    return BCOO((data, indices[idx_idx]), shape=data.shape)


def _spsp_pairdiff_impl(
    lhs_data: Tensor,
    lhs_indices: Tensor,
    rhs: TopKTensor
) -> Tuple[Tensor, Tensor]:
    lhs_data = jnp.broadcast_to(lhs_data, rhs.data.shape)
    lhs_indices = jnp.broadcast_to(lhs_indices, rhs.indices.shape)
    lhs = BCOO((lhs_data, lhs_indices), shape=rhs.shape)
    out = lhs - rhs
    return out.data, out.indices


def spsp_pairdiff(
    lhs: TopKTensor,
    rhs: TopKTensor,
) -> TopKTensor:
    """
    Pairwise difference between two top-k sparse matrices.
    """
    lhs_data = fold_and_promote(lhs.data, -2, lhs.data.shape[-2])
    lhs_indices = fold_and_promote(lhs.indices, -3, lhs.indices.shape[-3])
    data, indices = jax.vmap(
        partial(_spsp_pairdiff_impl, rhs=rhs),
        in_axes=(0, 0)
    )(lhs_data, lhs_indices)
    data = demote_and_unfold(data, -3, (-3,))
    indices = demote_and_unfold(indices, -4, (-4,))
    shape = lhs.shape[:-2] + (
        lhs.shape[-2], rhs.shape[-2], lhs.shape[-1])
    return BCOO((data, indices), shape=shape)


def random_sparse_batchfinal(key, shape, density=0.1):
    """
    Generate a random sparse matrix in batch-final COO format.
    """
    n = jnp.prod(jnp.array(shape))
    nse = int(density * n)
    k1, k2 = jax.random.split(key)
    indices = jax.random.choice(k1, a=n, shape=(nse,), replace=False)
    indices = jnp.stack(jnp.unravel_index(indices, shape), axis=-1)
    data = jax.random.normal(k2, (nse,))
    return BCOO((data, indices), shape=shape).sum_duplicates()


def to_batch_batchfinal(matrices: Sequence[Tensor]) -> Tensor:
    """
    Convert a sequence of sparse matrices to a batch of matrices using the
    batch-final, common-index COO format.

    .. note::
        This function is not intended to be compatible with JIT compilation.
    """
    batch_size = len(matrices)
    shape = sum([m.data.shape[0] for m in matrices])
    remaining_shape = matrices[0].data.shape[1:]
    indices = jnp.concatenate([m.indices for m in matrices], axis=0)
    data = jnp.zeros((shape, *remaining_shape, batch_size))
    start = 0
    for i, matrix in enumerate(matrices):
        end = start + matrix.data.shape[0]
        data = data.at[start:end, ..., i].set(matrix.data)
        start = end
    return BCOO(
        (data, indices),
        shape=(*matrices[0].shape, batch_size)
    ).sum_duplicates()


def _get_dense_dim_mm_batchfinal(lhs, rhs):
    """
    Get the dense dimension of the matrix multiplication.
    """
    lhs_dims = lhs.data.shape[1:]
    rhs_dims = rhs.data.shape[1:]
    # we don't check for broadcastability here
    return [max(l, r) for l, r in zip(lhs_dims, rhs_dims)]


def spspmm_batchfinal(lhs, rhs, inner_dims=(0, 0), outer_dims=(1, 1)):
    """
    Sparse-sparse matrix multiplication with vectorisation over trailing dense
    dimensions.

    .. note::
        This function is not recommended for use. It is maintained in case
        it is useful in the future. It is strongly recommended that you use
        the top-k format instead.
    """
    # only support 2D sparse for now
    assert lhs.n_sparse == rhs.n_sparse == 2
    dense_dim_out = _get_dense_dim_mm_batchfinal(lhs, rhs)
    out_shape = (
        lhs.shape[outer_dims[0]],
        rhs.shape[outer_dims[1]],
        *dense_dim_out
    )

    out_nse = lhs.nse * rhs.nse # memory use scales as product of NSEs
    lhs_data = lhs.data[None, ...]
    rhs_data = rhs.data[:, None, ...]

    lhs_contract_dim, rhs_contract_dim = inner_dims
    lhs_contract_idx = lhs.indices[:, lhs_contract_dim][None, :]
    rhs_contract_idx = rhs.indices[:, rhs_contract_dim][:, None]
    out_nonzero = (lhs_contract_idx == rhs_contract_idx)
    extra_idx = [None] * len(dense_dim_out)
    out_nonzero = out_nonzero[tuple([...] + extra_idx)]
    out_data = jnp.where(out_nonzero, lhs_data * rhs_data, 0.)

    lhs_indices = jnp.ones_like(lhs.indices).at[:, -2].set(
        lhs.indices[:, outer_dims[0]])
    rhs_indices = jnp.ones_like(rhs.indices).at[:, -1].set(
        rhs.indices[:, outer_dims[1]])
    out_indices = (lhs_indices[None, ...] * rhs_indices[:, None, ...])

    out_indices = out_indices.reshape(out_nse, -1)
    out_data = out_data.reshape(out_nse, *dense_dim_out)
    return BCOO((out_data, out_indices), shape=out_shape)


def embed_params_in_diagonal(params):
    dim = params.shape[-1]
    indices = jnp.arange(dim)
    indices = jnp.stack((indices, indices)).T[None, ...]
    idx_idx = tuple([None] * (params.ndim - 2) + [Ellipsis])
    return BCOO(
        (params, indices[idx_idx]), shape=(*params.shape[:-1], dim, dim)
    )


def embed_params_in_sparse(params):
    """
    Embed parameters in a sparse matrix.

    .. warning::
        This function is not intended to be compatible with JIT compilation.

    .. warning::
        The input must have at least 1 batch dimension. If it does not, a
        singleton batch dimension is added.
    """
    if params.ndim == 2:
        params = params[None, ...]
    dim = params.shape[-1]

    nzi = jax.lax.stop_gradient(jnp.abs(params))
    dims = list(range(params.ndim - 2))
    if dims:
        nzi = nzi.sum(dims)
    indices = jnp.stack(jnp.where(nzi))
    values = params[..., indices[0], indices[1]]
    indices = indices.T[None, ...]
    idx_idx = tuple([None] * (params.ndim - 3) + [Ellipsis])
    return BCOO(
        (values, indices[idx_idx]),
        shape=(*params.shape[:-2], dim, dim)
    )


def _promote_nnz_dim(values):
    return jnp.transpose(values, list(range(values.ndim))[::-1])
    # slightly faster but not as easy to implement
    #return jnp.transpose(values, (*list(range(values.ndim))[1:], 0))


def _demote_nnz_dim(values):
    return _promote_nnz_dim(values)
    #return jnp.transpose(values, (-1, *list(range(values.dim()))[:-1]))
