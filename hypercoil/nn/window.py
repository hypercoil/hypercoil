# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Window amplification
~~~~~~~~~~~~~~~~~~~~
Rebatching or channel amplification through data windowing.
"""
import jax
import equinox as eqx
from typing import Callable, Optional, Sequence, Union

from ..engine.paramutil import Tensor
from ..functional.window import sample_windows


class WindowAmplifier(eqx.Module):
    window_fn: Callable
    window_size: int
    augmentation_factor: int = 1
    augmentation_axis: int = 0
    windowing_axis: int = -1

    def __init__(
        self,
        window_size: int,
        allow_overlap: bool = False,
        create_new_axis: bool = False,
        augmentation_factor: int = 1,
        augmentation_axis: int = 0,
        windowing_axis: int = -1,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        self.window_fn = sample_windows(
            allow_overlap=allow_overlap,
            create_new_axis=create_new_axis,
        )
        self.window_size = window_size
        self.augmentation_factor = augmentation_factor
        self.augmentation_axis = augmentation_axis
        self.windowing_axis = windowing_axis

    def __call__(
        self,
        data: Union[Tensor, Sequence[Tensor]],
        split_key: bool = False,
        *,
        key: 'jax.random.PRNGKey',
    ):
        ## TODO: enable automask to exclude nan frames
        single_input = not isinstance(data, Sequence)
        if single_input:
            data = (data,)
        if split_key:
            keys = jax.random.split(key, len(data))
        else:
            keys = (key,) * len(data)
        out = tuple(self.window_fn(
            data,
            window_size=self.window_size,
            num_windows=self.augmentation_factor,
            windowing_axis=self.windowing_axis,
            multiplying_axis=self.augmentation_axis,
            key=key,
        ) for key, data in zip(keys, data))
        if single_input:
            out = out[0]
        return out
