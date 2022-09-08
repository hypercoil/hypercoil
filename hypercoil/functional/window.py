# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Windowing functions.
"""
import jax
import jax.numpy as jnp
import distrax
from typing import Callable, Generator, Sequence
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
    Sample non-overlapping windows from a tensor.

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


def _sample_nonoverlapping_impl(
    tensor: Tensor,
    window_size: int,
    select_fn: Callable,
    num_windows: int = 1,
    windowing_axis: int = -1,
    multiplying_axis: int = 0,
    *,
    key: 'jax.random.PRNGKey',
) -> Tensor:
    tensor = jnp.asarray(tensor)
    if windowing_axis < 0:
        windowing_axis = tensor.ndim + windowing_axis
    if multiplying_axis < 0:
        multiplying_axis = tensor.ndim + multiplying_axis
    input_size = tensor.shape[windowing_axis]
    unused_size = input_size - window_size * num_windows

    intervals = distrax.Multinomial(
        total_count=unused_size,
        probs=jnp.ones(num_windows + 1) / (num_windows + 1)
    ).sample(seed=key)
    start = jnp.arange(num_windows + 1) * window_size + jnp.cumsum(intervals)
    start = start[:-1]
    slices = [0] * tensor.ndim
    sizes = tuple(s if i != windowing_axis else window_size
                  for i, s in enumerate(tensor.shape))
    windows = tuple(select_fn(start, slices, sizes))
    return jnp.concatenate(windows, axis=multiplying_axis)


@document_window_fn
def sample_nonoverlapping_windows_existing_ax(
    tensor: Tensor,
    window_size: int,
    num_windows: int = 1,
    windowing_axis: int = -1,
    multiplying_axis: int = 0,
    *,
    key: 'jax.random.PRNGKey',
) -> Tensor:
    def _select_windows(
        start: Sequence,
        slices: Sequence,
        sizes: Sequence,
    ) -> Generator:
        for w in start:
            slc = slices.copy()
            slc[windowing_axis] = w
            window = jax.lax.dynamic_slice(tensor, tuple(slc), sizes)
            yield window

    return _sample_nonoverlapping_impl(
        tensor,
        window_size=window_size,
        select_fn=_select_windows,
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
    key: 'jax.random.PRNGKey',
) -> Tensor:
    def _select_windows(start, slices, sizes):
        for w in start:
            slc = slices.copy()
            slc[windowing_axis] = w
            window = jax.lax.dynamic_slice(tensor, tuple(slc), sizes)
            window = jnp.expand_dims(window, axis=multiplying_axis)
            yield window

    return _sample_nonoverlapping_impl(
        tensor,
        window_size=window_size,
        select_fn=_select_windows,
        num_windows=num_windows,
        windowing_axis=windowing_axis,
        multiplying_axis=multiplying_axis,
        key=key,
    )


def sample_nonoverlapping_windows(
    create_new_axis: bool = False,
) -> Callable:
    """
    Sample non-overlapping windows from a tensor.

    Parameters
    ----------
    create_new_axis : bool, optional
        Whether to create a new axis for the windows. Default: False.
        If this is True, the new axis will be inserted at
        ``multiplying_axis``. Otherwise, the windows will be multiplied along
        the existing ``multiplying_axis``.

    Returns
    -------
    callable
        A function that samples non-overlapping windows from a tensor.
    """
    if create_new_axis:
        return sample_nonoverlapping_windows_new_ax
    return sample_nonoverlapping_windows_existing_ax
