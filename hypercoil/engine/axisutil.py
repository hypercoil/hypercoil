# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Miscellaneous utility functions for tensor axis manipulation.
"""
import jax
import jax.numpy as jnp
from functools import partial, reduce
from typing import Any, Callable, Generator, Tuple, Union
from jax import vmap
from jax.tree_util import tree_map, tree_reduce
from .paramutil import PyTree, Tensor


def _dim_or_none(x, i):
    proposal = i + x
    if proposal < 0:
        return None
    return proposal


def _compose(
    f: Any,
    g: Callable,
) -> Any:
    return g(f)


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


#TODO: use chex to evaluate how often this has to compile when using
#      jit + vmap_over_outer
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
        _compose,
        #lambda x, g: g(x),
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
    if axis < 0:
        axis += ndim
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


# def _ax_out(ax, compl, ax_idx, compl_idx, out_idx):
#     return (out_idx + 1, ax_idx + 1, compl_idx), ax[ax_idx]


# def _compl_out(ax, compl, ax_idx, compl_idx, out_idx):
#     return (out_idx + 1, ax_idx, compl_idx + 1), compl[compl_idx]


# def _demote_ax_impl(carry, pred, out, ax, compl) -> Tensor:
#     out_idx, ax_idx, compl_idx = carry
#     #pred = (out_idx in out)
#     carry, ret = jax.lax.cond(
#         pred, _ax_out, _compl_out,
#         ax, compl, ax_idx, compl_idx, out_idx)
#     return carry, ret


# def demote_axis(
#     ndim: int,
#     axis: Union[int, Tuple[int, ...]],
# ) -> Tuple[int, ...]:
#     """
#     Demote the outermost axis or axes to the specified dimension(s).
#     """
#     if isinstance(axis, int): axis = (axis,)
#     out = [standard_axis_number(ax, ndim) for ax in axis]
#     ax = jnp.arange(len(axis))
#     compl = jnp.arange(len(axis), ndim)
#     pred = (jnp.array(out) == jnp.arange(ndim)[..., None]).any(-1)
#     _, ax = jax.lax.scan(
#         partial(_demote_ax_impl, out=out, ax=ax, compl=compl),
#         init=(0, 0, 0),
#         xs=pred,
#         length=ndim
#     )
#     return ax


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
    # shape = jnp.array(tensor.shape)
    # ndim = len(shape)
    # ax = jnp.array(axes)[..., None]
    # pred = (jnp.arange(ndim) == ax).any(0)
    # #prod = jnp.prod(jnp.where(pred, shape, 1)).item()
    # print(pred, shape)
    # prod, _ = jax.lax.scan(
    #     partial(_reduce_cond, f=_prod, identity=_id_mul),
    #     1,
    #     (tensor.shape, pred))
    # print(prod)
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
