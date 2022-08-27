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
import numpy as np
import distrax
from jax import vmap
from jax.tree_util import tree_map, tree_reduce
from jax.experimental.sparse import BCOO
from functools import partial, reduce
from typing import Any, Callable, Generator, Optional, Sequence, Tuple, Union


#TODO: replace with jaxtyping at some point
Tensor = Union[jnp.DeviceArray, np.ndarray]
PyTree = Any
Distribution = distrax.Distribution


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


def _compose(
    f: Any,
    g: Callable,
) -> Any:
    return g(f)


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


def sample_multivariate(
    *,
    distr: Distribution,
    shape: Tuple[int],
    event_axes: Sequence[int],
    mean_correction: bool = False,
    key: jax.random.PRNGKey
):
    ndim = len(shape)
    event_axes = tuple(
        [standard_axis_number(axis, ndim) for axis in event_axes])
    event_shape = tuple([shape[axis] for axis in event_axes])
    sample_shape = tuple([shape[axis] for axis in range(ndim)
                          if axis not in event_axes])

    # This doesn't play well with JIT compilation.
    # if distr.event_shape != event_shape:
    #     raise ValueError(
    #         f"Distribution event shape {distr.event_shape} does not match "
    #         f"tensor shape {shape} along axes {event_axes}."
    #     )
    val = distr.sample(seed=key, sample_shape=sample_shape)

    if mean_correction:
        try:
            correction = 1 / (distr.mean() + jnp.finfo(jnp.float32).eps)
        except AttributeError:
            correction = 1 / (val.mean() + jnp.finfo(jnp.float64).eps)
        val = val * correction

    axis_order = argsort(axis_complement(ndim, event_axes) + event_axes)
    return jnp.transpose(val, axes=axis_order)


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
