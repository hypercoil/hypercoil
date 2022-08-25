# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Generic image / preimage mapper.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Optional, Tuple
from ..functional.utils import (
    Tensor, PyTree
)


class MappedParameter(eqx.Module):
    param_name: str = "weight"
    original: Tensor

    def __init__(self, model: PyTree, *, param_name: str = "weight"):
        self.param_name = param_name
        self.original = self.preimage_map(
            model.__getattribute__(param_name))

    def preimage_map(self, param: Tensor) -> Tensor:
        raise NotImplementedError()

    def image_map(self, param: Tensor) -> Tensor:
        raise NotImplementedError()

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


class OutOfDomainHandler(eqx.Module):
    def test(self, x: Tensor, bound: Tuple[float, float]) -> Tensor:
        return jnp.logical_and(
            x <= bound[-1],
            x >= bound[0]
        )


class Clip(OutOfDomainHandler):
    def apply(self, x: Tensor, bound: Tuple[float, float]) -> Tensor:
        x = jax.lax.stop_gradient(x)
        return jnp.clip(x, bound[0], bound[-1])


class Renormalise(OutOfDomainHandler):
    def apply(
        self,
        x: Tensor,
        bound: Tuple[float, float],
        axis: Optional[Tuple[int, ...]] = None
    ) -> Tensor:
        x = jax.lax.stop_gradient(x)
        upper = x.max(axis)
        lower = x.min(axis)
        unew = jnp.minimum(bound[-1], upper)
        lnew = jnp.maximum(bound[0], lower)
        out = x - x.mean(axis)
        out = out / ((upper - lower) / (unew - lnew))
        return out + lnew - out.min(axis)
