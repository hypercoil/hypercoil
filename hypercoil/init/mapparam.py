# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Generic image / preimage mapper.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
from typing import Any, Callable, Optional, Tuple
from ..functional.utils import (
    Tensor, PyTree, complex_decompose, complex_recompose
)


class MappedParameter(eqx.Module):
    original: Tensor
    param_name: str = "weight"

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


class DomainMappedParameter(MappedParameter):
    original: Tensor
    param_name: str = "weight"
    image_bound: Any = None
    preimage_bound: Any = None
    handler: Any = None

    def __init__(
        self,
        model: PyTree,
        *,
        param_name: str = "weight",
        image_bound: Any = None,
        preimage_bound: Any = None,
        handler: Callable = None
    ):
        self.handler = handler or Clip()
        self.image_bound = image_bound or (-float('inf'), float('inf'))
        self.preimage_bound = preimage_bound or (-float('inf'), float('inf'))
        super(DomainMappedParameter, self).__init__(
            model=model, param_name=param_name)

    def preimage_map(self, param: Tensor) -> Tensor:
        x = self.handler.apply(param, self.image_bound)
        i = self.preimage_map_impl(x)
        i = self.handler.apply(i, self.preimage_bound)
        return i

    def image_map(self, param: Tensor) -> Tensor:
        return self.image_map_impl(param)

    def preimage_map_impl(self, param: Tensor) -> Tensor:
        raise NotImplementedError()

    def image_map_impl(self, param: Tensor) -> Tensor:
        raise NotImplementedError()

    def test(self, param: Tensor):
        return self.handler.test(param, self.image_bound)

    def handle_ood(self, param: Tensor) -> Tensor:
        return self.handler.apply(param, self.image_bound)


class AffineDomainMappedParameter(DomainMappedParameter):
    original: Tensor
    param_name: str = "weight"
    loc: Tensor = 0.
    scale: Tensor = 1.
    image_bound: Any = None
    preimage_bound: Any = None
    handler: Any = None

    def __init__(
        self,
        model: PyTree,
        *,
        loc: Tensor = 0.,
        scale: Tensor = 1.,
        param_name: str = "weight",
        image_bound: Any = None,
        preimage_bound: Any = None,
        handler: Callable = None
    ):
        self.loc = loc
        self.scale = scale
        super(AffineDomainMappedParameter, self).__init__(
            model=model,
            param_name=param_name,
            image_bound=image_bound,
            preimage_bound=preimage_bound,
            handler=handler
        )

    def preimage_map(self, param: Tensor) -> Tensor:
        x = self.handler.apply(param, self.image_bound)
        i = self.preimage_map_impl((x - self.loc) / self.scale)
        i = self.handler.apply(i, self.preimage_bound)
        return i

    def image_map(self, param: Tensor) -> Tensor:
        return self.scale * self.image_map_impl(param) + self.loc


class PhaseAmplitudeMixin:
    def preimage_map(self, param: Tensor) -> Tensor:
        ampl, phase = complex_decompose(param)
        ampl = super().preimage_map(ampl)
        return complex_recompose(ampl, phase)

    def image_map(self, param: Tensor) -> Tensor:
        ampl, phase = complex_decompose(param)
        ampl = super().image_map(ampl)
        return complex_recompose(ampl, phase)


class IdentityMappedParameter(MappedParameter):
    def preimage_map(self, param: Tensor) -> Tensor:
        return param

    def image_map(self, param: Tensor) -> Tensor:
        return param


class AffineMappedParameter(AffineDomainMappedParameter):

    def preimage_map_impl(self, param: Tensor) -> Tensor:
        return param

    def image_map_impl(self, param: Tensor) -> Tensor:
        return param


class TanhMappedParameter(AffineDomainMappedParameter):
    def __init__(
        self,
        model: PyTree,
        *,
        param_name: str = "weight",
        preimage_bound: Tuple[float, float] = (-3., 3.),
        handler: Callable = None,
        scale: float = 1.,
    ):
        super().__init__(
            model,
            param_name=param_name,
            preimage_bound=preimage_bound,
            image_bound=(-scale, scale),
            handler=handler,
            scale=scale,
        )

    def preimage_map_impl(self, param: Tensor) -> Tensor:
        return jnp.arctanh(param)

    def image_map_impl(self, param: Tensor) -> Tensor:
        return jnp.tanh(param)


class AmplitudeTanhMappedParameter(PhaseAmplitudeMixin, TanhMappedParameter):
    pass
