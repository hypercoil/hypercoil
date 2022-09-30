# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Windowing functions.
"""
from __future__ import annotations
from typing import Callable, Generator, Sequence

import jax
import jax.numpy as jnp
import distrax

from ..engine import Tensor


def document_window_fn(fn: Callable) -> Callable:
    """
    Decorator to document a windowing function.

    Parameters
    ----------
    fn : callable
        The function to document.

    Returns
    -------
    callable
        The decorated function.
    """
    fn.__doc__ = f"""
    Sample windows from a tensor.

    Parameters
    ----------
    tensor : Tensor
        The tensor to sample from.
    window_size : int
        The size of the window(s) to sample.
    num_windows : int, optional
        The number of windows to sample. Default: 1.
    windowing_axis : int, optional
        The axis along which to sample windows. Default: -1.
    multiplying_axis : int, optional
        The axis along which to multiply the windows. Default: 0.

    Returns
    -------
    Tensor
        Windows from the input tensor stacked along the multiplying axis.
    """
    return fn


def _select_fn_allow_overlap(
    input_size: int,
    window_size: int,
    num_windows: int,
    key: "jax.random.PRNGKey",
) -> Tensor:
    return jax.random.choice(
        key,
        a=(input_size - window_size + 1),
        shape=(num_windows,),
        replace=False,
    )


def _select_fn_no_overlap(
    input_size: int,
    window_size: int,
    num_windows: int,
    key: "jax.random.PRNGKey",
) -> Tensor:
    unused_size = input_size - window_size * num_windows
    intervals = distrax.Multinomial(
        total_count=unused_size,
        probs=jnp.ones(num_windows + 1) / (num_windows + 1),
    ).sample(seed=key)
    start = jnp.arange(num_windows + 1) * window_size + jnp.cumsum(intervals)
    return start[:-1]


def _slice_fn_existing_ax(
    tensor: Tensor,
    start: Sequence,
    slices: Sequence,
    sizes: Sequence,
    windowing_axis: int = -1,
    multiplying_axis: int = 0,
) -> Generator:
    for w in start:
        slc = slices.copy()
        slc[windowing_axis] = w
        window = jax.lax.dynamic_slice(tensor, tuple(slc), sizes)
        yield window


def _slice_fn_new_ax(
    tensor: Tensor,
    start: Sequence,
    slices: Sequence,
    sizes: Sequence,
    windowing_axis: int = -1,
    multiplying_axis: int = 0,
) -> Generator:
    for w in start:
        slc = slices.copy()
        slc[windowing_axis] = w
        window = jax.lax.dynamic_slice(tensor, tuple(slc), sizes)
        window = jnp.expand_dims(window, axis=multiplying_axis)
        yield window


def _sample_window_impl(
    tensor: Tensor,
    window_size: int,
    slice_fn: Callable,
    select_fn: Callable,
    num_windows: int = 1,
    windowing_axis: int = -1,
    multiplying_axis: int = 0,
    *,
    key: "jax.random.PRNGKey",
) -> Tensor:
    tensor = jnp.asarray(tensor)
    if windowing_axis < 0:
        windowing_axis = tensor.ndim + windowing_axis
    if multiplying_axis < 0:
        multiplying_axis = tensor.ndim + multiplying_axis
    input_size = tensor.shape[windowing_axis]

    start = select_fn(
        input_size=input_size,
        window_size=window_size,
        num_windows=num_windows,
        key=key,
    )
    slices = [0] * tensor.ndim
    sizes = tuple(
        s if i != windowing_axis else window_size
        for i, s in enumerate(tensor.shape)
    )
    windows = tuple(
        slice_fn(
            tensor=tensor,
            start=start,
            slices=slices,
            sizes=sizes,
            windowing_axis=windowing_axis,
            multiplying_axis=multiplying_axis,
        )
    )
    return jnp.concatenate(windows, axis=multiplying_axis)


@document_window_fn
def sample_nonoverlapping_windows_existing_ax(
    tensor: Tensor,
    window_size: int,
    num_windows: int = 1,
    windowing_axis: int = -1,
    multiplying_axis: int = 0,
    *,
    key: "jax.random.PRNGKey",
) -> Tensor:
    return _sample_window_impl(
        tensor,
        window_size=window_size,
        slice_fn=_slice_fn_existing_ax,
        select_fn=_select_fn_no_overlap,
        num_windows=num_windows,
        windowing_axis=windowing_axis,
        multiplying_axis=multiplying_axis,
        key=key,
    )


@document_window_fn
def sample_nonoverlapping_windows_new_ax(
    tensor: Tensor,
    window_size: int,
    num_windows: int = 1,
    windowing_axis: int = -1,
    multiplying_axis: int = 0,
    *,
    key: "jax.random.PRNGKey",
) -> Tensor:
    return _sample_window_impl(
        tensor,
        window_size=window_size,
        slice_fn=_slice_fn_new_ax,
        select_fn=_select_fn_no_overlap,
        num_windows=num_windows,
        windowing_axis=windowing_axis,
        multiplying_axis=multiplying_axis,
        key=key,
    )


@document_window_fn
def sample_overlapping_windows_existing_ax(
    tensor: Tensor,
    window_size: int,
    num_windows: int = 1,
    windowing_axis: int = -1,
    multiplying_axis: int = 0,
    *,
    key: "jax.random.PRNGKey",
) -> Tensor:
    return _sample_window_impl(
        tensor,
        window_size=window_size,
        slice_fn=_slice_fn_existing_ax,
        select_fn=_select_fn_allow_overlap,
        num_windows=num_windows,
        windowing_axis=windowing_axis,
        multiplying_axis=multiplying_axis,
        key=key,
    )


@document_window_fn
def sample_overlapping_windows_new_ax(
    tensor: Tensor,
    window_size: int,
    num_windows: int = 1,
    windowing_axis: int = -1,
    multiplying_axis: int = 0,
    *,
    key: "jax.random.PRNGKey",
) -> Tensor:
    return _sample_window_impl(
        tensor,
        window_size=window_size,
        slice_fn=_slice_fn_new_ax,
        select_fn=_select_fn_allow_overlap,
        num_windows=num_windows,
        windowing_axis=windowing_axis,
        multiplying_axis=multiplying_axis,
        key=key,
    )


def sample_windows(
    allow_overlap: bool = False,
    create_new_axis: bool = False,
) -> Callable:
    """
    Sample windows from a tensor.

    Parameters
    ----------
    allow_overlap : bool (default: False)
        Whether to allow windows to overlap.
    create_new_axis : bool (default: False)
        Whether to create a new axis for the windows. Default: False.
        If this is True, the new axis will be inserted at
        ``multiplying_axis``. Otherwise, the windows will be multiplied along
        the existing ``multiplying_axis``.

    Returns
    -------
    callable
        A function that samples windows from a tensor.
    """
    if create_new_axis:
        if allow_overlap:
            return sample_overlapping_windows_new_ax
        return sample_nonoverlapping_windows_new_ax
    if allow_overlap:
        return sample_overlapping_windows_existing_ax
    return sample_nonoverlapping_windows_existing_ax
