# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Generic image / preimage mapper.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
from hypercoil.functional.utils import Tensor, PyTree


class MappedParameter(eqx.Module):
    param_name: str = "weight"
    original: Tensor

    def __init__(self, model: PyTree, *, param_name: str = "weight"):
        self.param_name = param_name
        self.original = self.preimage_map(
            model.__getattribute__(param_name))

    def preimage_map(self, param: Tensor) -> Tensor:
        return jnp.log(param)

    def image_map(self, param: Tensor) -> Tensor:
        return jax.nn.softmax(param, axis=-1)

    def __jax_array__(self):
        return self.image_map(self.original)

    @classmethod
    def embed(
        cls,
        model: PyTree,
        *pparams,
        param_name: str = "weight",
        **params
    ):
        mapped = cls(model=model, *pparams, param_name=param_name, **params)
        return eqx.tree_at(
            lambda m: m.__getattribute__(mapped.param_name),
            model,
            replace=mapped
        )
