# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for windowing module.
"""
from posixpath import split
import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
from hypercoil.nn.window import WindowAmplifier


class TestWindow:
    def test_window(self):
        key = jax.random.PRNGKey(0)

        input_multi = jnp.arange(1000)
        input_multi = jnp.stack((
            input_multi, input_multi, input_multi
        ), axis=0)
        input_single = jnp.arange(1000)
        window_fn = eqx.filter_jit(
            WindowAmplifier(
                allow_overlap=False,
                create_new_axis=True,
                window_size=90,
                augmentation_factor=10
            ),
        )

        # Use the same windows for both inputs. This can be useful when
        # windowing both BOLD and confounds, for example.
        out_multi, out_single = window_fn(
            (input_multi, input_single),
            key=key,
        )
        for i in range(10):
            assert (out_multi[i, 0] == out_single[i]).all()

        # Use different windows for each input.
        out_multi, out_single = window_fn(
            (input_multi, input_single),
            split_key=True,
            key=key,
        )
        for i in range(10):
            # Well, there is a chance this would fail, but it's inordinately
            # unlikely. And, for this random seed, it doesn't.
            assert not (out_multi[i, 0] == out_single[i]).all()
