# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Miscellaneous utility functions for neural network parameters.
"""
from __future__ import annotations
from typing import Any, Union

import jax
import numpy as np
from numpyro.distributions import Distribution as Distr


# TODO: replace with jaxtyping at some point
Tensor = Union[jax.Array, np.ndarray]
PyTree = Any
Distribution = Distr


# From ``equinox``:
# TODO: remove this once JAX fixes the issue.
# Working around JAX not always properly respecting __jax_array__ . . .
# See JAX issue #10065
def _to_jax_array(param: Tensor) -> Tensor:
    if hasattr(param, '__jax_array__'):
        return param.__jax_array__()
    else:
        return param


def where_weight(model: PyTree) -> Tensor:
    return model.weight
