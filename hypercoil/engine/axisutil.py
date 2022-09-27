# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Miscellaneous utility functions for tensor axis manipulation.
"""
import jax
import jax.numpy as jnp
from functools import partial, reduce
from typing import Any, Callable, Generator, Optional, Sequence, Tuple, Union
from jax import vmap
from jax.tree_util import tree_map, tree_reduce
from .paramutil import PyTree, Tensor


def _dim_or_none(x, align, i, ndmax):
    if not align:
        i = i - ndmax
    else:
        i = -i
    proposal = i + x
    if proposal < 0:
        return None
    elif align:
        return -i
    return proposal


def _compose(
    f: Any,
    g: Callable,
) -> Any:
    return g(f)


def _seq_pad(
    x: Tuple[Any, ...],
    n: int,
    pad: str = 'last',
    pad_value: Any = None,
) -> Tuple[Any, ...]:
    padding = [pad_value for _ in range(n + 1 - len(x))]
    if pad == 'last':
        return tuple((*x, *padding))
    elif pad == 'first':
        return tuple((*padding, *x))
    raise ValueError(f"Invalid padding: {pad}")


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


def broadcast_ignoring(
    x: Tensor,
    y: Tensor,
    axis: Union[int, Tuple[int, ...]],
) -> Tensor:
    """
    Broadcast two tensors, ignoring the axis or axes specified.

    This can be useful, for instance, when concatenating tensors along
    the ignored axis.
    """
    def _form_reduced_shape(axes, shape, ndim):
        axes = tuple(standard_axis_number(a, ndim) for a in axes)
        shape_reduced = tuple(1 if i in axes else shape[i]
                              for i in range(ndim))
        return shape_reduced, axes

    def _form_final_shape(axes_out, axes_in, shape_in, common_shape):
        j = 0
        for i, s in enumerate(common_shape):
            if i not in axes_out:
                yield s
            else:
                ax = axes_in[j]
                if ax is None or ax > len(shape_in):
                    yield 1
                else:
                    yield shape_in[ax]
                j += 1

    if isinstance(axis, int): axis = (axis,)
    shape_x, shape_y = x.shape, y.shape
    shape_x_reduced, axes_x = _form_reduced_shape(axis, shape_x, x.ndim)
    shape_y_reduced, axes_y = _form_reduced_shape(axis, shape_y, y.ndim)
    common_shape = jnp.broadcast_shapes(shape_x_reduced, shape_y_reduced)
    axes_out = tuple(standard_axis_number(a, len(common_shape)) for a in axis)
    shape_y = tuple(_form_final_shape(
        axes_out=axes_out,
        axes_in=axes_y,
        shape_in=shape_y,
        common_shape=common_shape)
    )
    shape_x = tuple(_form_final_shape(
        axes_out=axes_out,
        axes_in=axes_x,
        shape_in=shape_x,
        common_shape=common_shape)
    )
    return jnp.broadcast_to(x, shape_x), jnp.broadcast_to(y, shape_y)


#TODO: use chex to evaluate how often this has to compile when using
#      jit + vmap_over_outer
def apply_vmap_over_outer(
    x: PyTree,
    f: Callable,
    f_dim: int,
    align_outer: bool = False,
    structuring_arg: Optional[Union[Callable, int]] = None,
) -> Tensor:
    """
    Apply a function across the outer dimensions of a tensor.
    """
    if isinstance(f_dim, int):
        f_dim = tree_map(lambda _: f_dim, x)
    if isinstance(align_outer, bool):
        align_outer = tree_map(lambda _: align_outer, x)
    ndim = tree_map(lambda x, f: x.ndim - f - 1, x, f_dim)
    ndmax = tree_reduce(max, ndim)
    if structuring_arg is None:
        output_structure = range(0, ndmax + 1)
    else:
        if isinstance(structuring_arg, int):
            output_structure = range(
                0, x[structuring_arg].ndim - f_dim[structuring_arg])
            criterion = align_outer[structuring_arg]
        else:
            output_structure = range(
                0, structuring_arg(x).ndim - structuring_arg(f_dim))
            criterion = structuring_arg(align_outer)
        if criterion:
            output_structure = _seq_pad(output_structure, ndmax, 'last')
        else:
            output_structure = _seq_pad(output_structure, ndmax, 'first')
    # print(ndim, tuple(range(ndmax + 1)))
    # print([(
    #    tree_map(
    #         partial(_dim_or_none, i=i, ndmax=ndmax),
    #         ndim,
    #         align_outer
    #     ), i, o)
    #     for i, o in zip(range(0, ndmax + 1), output_structure)
    # ])
    return reduce(
        _compose,
        #lambda x, g: g(x),
        [partial(
            vmap,
            in_axes=tree_map(
                partial(_dim_or_none, i=i, ndmax=ndmax),
                ndim,
                align_outer
            ),
            out_axes=o
        ) for i, o in zip(range(0, ndmax + 1), output_structure)],
        f
    )(*x)


def vmap_over_outer(
    f: Callable,
    f_dim: int,
    align_outer: bool = False,
    structuring_arg: Optional[Union[Callable, int]] = None,
) -> Callable:
    """
    Transform a function to apply to the outer dimensions of a tensor.
    """
    return partial(
        apply_vmap_over_outer,
        f=f,
        f_dim=f_dim,
        align_outer=align_outer,
        structuring_arg=structuring_arg,
    )


def axis_complement(
    ndim: int,
    axis: Union[int, Tuple[int, ...]],
) -> Tuple[int, ...]:
    """
    Return the complement of the axis or axes for a tensor of dimension ndim.
    """
    if isinstance(axis, int): axis = (axis,)
    ax = [True for _ in range(ndim)]
    for a in axis:
        ax[a] = False
    ax = [i for i, a in enumerate(ax) if a]
    return tuple(ax)


def standard_axis_number(axis: int, ndim: int) -> int:
    """
    Convert an axis number to a standard axis number.
    """
    if axis < 0 and axis >= -ndim:
        axis += ndim
    elif axis < -ndim or axis >= ndim:
        return None
    return axis


def negative_axis_number(axis: int, ndim: int) -> int:
    """
    Convert a standard axis number to a negative axis number.
    """
    if axis >= 0:
        axis -= ndim
    return axis


def promote_axis(
    ndim: int,
    axis: Union[int, Tuple[int, ...]],
) -> Tuple[int, ...]:
    """
    Promote an axis or axes to the outermost dimension.
    """
    if isinstance(axis, int): axis = (axis,)
    axis = [standard_axis_number(ax, ndim) for ax in axis]
    return (*axis, *axis_complement(ndim, axis))


def _demote_axis(
    ndim: int,
    axis: Union[int, Tuple[int, ...]],
) -> Generator:
    """Helper function for axis demotion."""
    compl = range(len(axis), ndim).__iter__()
    src = range(len(axis)).__iter__()
    for ax in range(ndim):
        if ax in axis:
            yield src.__next__()
        else:
            yield compl.__next__()


def demote_axis(
    ndim: int,
    axis: Union[int, Tuple[int, ...]],
) -> Tuple[int, ...]:
    if isinstance(axis, int): axis = (axis,)
    axis = [standard_axis_number(ax, ndim) for ax in axis]
    return tuple(_demote_axis(ndim=ndim, axis=axis))


@partial(jax.jit, static_argnames=('axis', 'n_folds'))
def fold_axis(tensor: Tensor, axis: int, n_folds: int) -> Tensor:
    """
    Fold the specified axis into the specified number of folds.
    """
    axis = standard_axis_number(axis, tensor.ndim)
    shape = tensor.shape
    current = shape[axis]
    new_shape = (
        shape[:axis] +
        (current // n_folds, n_folds) +
        shape[axis + 1:]
    )
    return tensor.reshape(new_shape)


# Apparently lambdas will give us trouble with the compiler.
# So we have this trash instead.
def _prod(x, y): return x * y
def _sum(x, y): return x + y
def _left(x, y): return x
def _right(x, y): return y
def _noop(x): return x
def _id_mul(x): return 1
def _id_add(x): return 0

def _reduce_cond(acc, x, f, identity):
    data, pred = x
    nxt = jax.lax.cond(pred, _noop, identity, data)
    acc = f(acc, nxt)
    return acc, None


@partial(jax.jit, static_argnames=('axes',))
def unfold_axes(tensor: Tensor, axes: Union[int, Tuple[int, ...]]) -> Tensor:
    """
    Unfold the specified consecutive axes into a single new axis.

    This function will fail if the specified axes are not consecutive.
    """
    if isinstance(axes, int):
        return tensor
    shape = tensor.shape
    axes = [standard_axis_number(ax, tensor.ndim) for ax in axes]
    current = [shape[ax] for ax in axes]
    prod = reduce(_prod, current)
    new_shape = (
        tensor.shape[:axes[0]] +
        (prod,) +
        tensor.shape[axes[-1] + 1:]
    )
    return tensor.reshape(new_shape)


@partial(jax.jit, static_argnames=('axis', 'n_folds'))
def fold_and_promote(tensor: Tensor, axis: int, n_folds: int) -> Tensor:
    """
    Fold the specified axis into the specified number of folds, and promote
    the new axis across the number of folds to the outermost dimension.
    """
    folded = fold_axis(tensor, axis, n_folds)
    return jnp.transpose(folded, promote_axis(folded.ndim, axis))


@partial(jax.jit, static_argnames=('target_address', 'axes'))
def demote_and_unfold(
    tensor: Tensor,
    target_address: int,
    axes: Union[int, Tuple[int, ...]]
):
    demoted = jnp.transpose(tensor, demote_axis(tensor.ndim, target_address))
    return unfold_axes(demoted, axes)


def argsort(seq):
    # Sources:
    # (1) https://stackoverflow.com/questions/3382352/ ...
    #     equivalent-of-numpy-argsort-in-basic-python
    # (2) http://stackoverflow.com/questions/3071415/ ...
    #     efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


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
