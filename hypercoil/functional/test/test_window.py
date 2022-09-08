# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for windowing operations.
"""
import jax
import jax.numpy as jnp
import numpy as np
from hypercoil.functional.window import (
    sample_windows,
)


class TestWindow:
    def test_window_nonoverlapping(self):
        key = jax.random.PRNGKey(0)

        input = jnp.arange(1000).reshape(10, 100)
        window_fn = jax.jit(
            sample_windows(False, True),
            static_argnames=('window_size', 'num_windows')
        )
        out = window_fn(
            input,
            window_size=10,
            num_windows=10,
            key=key
        )
        ref = jnp.stack(jnp.split(input, 10, axis=1))
        assert out.shape == (10, 10, 10)
        assert (out == ref).all()

        for size in (50, 70, 90):
            input = jnp.arange(1000)[None, :]
            window_fn = jax.jit(
                sample_windows(False, False),
                static_argnames=('window_size', 'num_windows')
            )
            out = window_fn(
                input,
                window_size=size,
                num_windows=10,
                key=key
            )
            out = np.array(out)
            assert out.shape == (10, size)
            for i, window in enumerate(out):
                for j in range(i + 1, len(out)):
                    diff = np.abs(window - out[[j]].T)
                    assert (diff > 0).all()

    def test_window_overlapping(self):
        key = jax.random.PRNGKey(0)

        input = jnp.arange(1000).reshape(10, 100)
        window_fn = jax.jit(
            sample_windows(True, True),
            static_argnames=('window_size', 'num_windows')
        )
        out = window_fn(
            input,
            window_size=10,
            num_windows=10,
            key=key
        )
        assert out.shape == (10, 10, 10)

        input = jnp.arange(1000)[None, :]
        window_fn = jax.jit(
            sample_windows(True, False),
            static_argnames=('window_size', 'num_windows')
        )
        out = window_fn(
            input,
            window_size=50,
            num_windows=10,
            key=key
        )
        out = np.array(out)
        assert out.shape == (10, 50)
        a = 0
        for i, window in enumerate(out):
            for j in range(i + 1, len(out)):
                diff = np.abs(window - out[[j]].T)
                a += (diff == 0).sum()
        # some windows overlap. Not guaranteed, but works for this seed.
        assert a > 0
