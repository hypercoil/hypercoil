# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Parameter mappers / mapped parameters for ``equinox`` modules.
Similar to PyTorch's ``torch.nn.utils.parametrize``.
"""
import equinox as eqx
import jax
import jax.numpy as jnp
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union
from ..functional.activation import isochor
from ..functional.matrix import spd
from ..functional.utils import (
    Tensor, PyTree, complex_decompose, complex_recompose
)


# From ``equinox``:
# TODO: remove this once JAX fixes the issue.
# Working around JAX not always properly respecting __jax_array__ . . .
# See JAX issue #10065
def _to_jax_array(param: Tensor) -> Tensor:
    if hasattr(param, "__jax_array__"):
        return param.__jax_array__()
    else:
        return param


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


class ForcePositiveDefinite(OutOfDomainHandler):
    spd_threshold: float = 1e-6

    def __init__(self, spd_threshold:float = 1e-6):
        self.spd_threshold = spd_threshold
        super().__init__()

    def apply(
        self,
        x: Tensor,
        bound: None = None,
        eps: Optional[float] = None,
    ) -> Tensor:
        if eps is None: eps = self.spd_threshold
        x = jax.lax.stop_gradient(x)
        return spd(x, eps=eps)

    def test(self, *pparams, **params):
        raise NotImplementedError()


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


class MappedLogits(AffineDomainMappedParameter):
    def __init__(
        self,
        model: PyTree,
        *,
        param_name: str = "weight",
        preimage_bound: Tuple[float, float] = (-4.5, 4.5),
        handler: Callable = None,
        loc: Optional[float] = None,
        scale: float = 1.,
    ):
        if loc is None: loc = (scale / 2)
        shift = loc - scale / 2
        super().__init__(
            model,
            param_name=param_name,
            preimage_bound=preimage_bound,
            image_bound=(loc - scale / 2, loc + scale / 2),
            handler=handler,
            loc=shift,
            scale=scale,
        )

    def preimage_map_impl(self, param: Tensor) -> Tensor:
        return jax.scipy.special.logit(param)

    def image_map_impl(self, param: Tensor) -> Tensor:
        return jax.nn.sigmoid(param)


class NormSphereParameter(AffineDomainMappedParameter):
    normalise_fn: Callable
    order: Tensor = 2
    axis: Union[int, Tuple[int, ...]] = -1

    def __init__(
        self,
        model: PyTree,
        *,
        param_name: str = "weight",
        handler: Callable = None,
        loc: float = 0.,
        scale: float = 1.,
        norm: float = 2.,
        axis: Union[int, Tuple[int, ...]] = -1,
    ):
        self.order = norm
        self.axis = axis
        if isinstance(norm, jnp.ndarray):
            def ellipse_norm(x, **params):
                x = x.swapaxes(-1, axis)
                norms = x[..., None, :] @ norm @ x[..., None]
                return jnp.sqrt(norms).squeeze(-1).swapaxes(-1, axis)
            f = ellipse_norm
        else:
            f = jnp.linalg.norm
        def normalise(x):
            n = f(
                x,
                ord=norm,
                axis=axis,
                keepdims=True
            ) + jnp.finfo(x.dtype).eps
            return x / n

        self.normalise_fn = normalise

        super().__init__(
            model, param_name=param_name,
            handler=handler, loc=loc, scale=scale
        )

    def preimage_map_impl(self, param: Tensor) -> Tensor:
        return param

    def image_map_impl(self, param: Tensor) -> Tensor:
        return self.normalise_fn(param)


def paramlog(x: Tensor, temperature: float, smoothing: float) -> Tensor:
    return temperature * jnp.log(x + smoothing)


def paramsoftmax(
    x: Tensor,
    temperature: float,
    axis: Union[int, Tuple[int, ...]]
) -> Tensor:
    return jax.nn.softmax(x / temperature, axis)


class ProbabilitySimplexParameter(DomainMappedParameter):
    _image_map_impl: Callable
    _preimage_map_impl: Callable
    axis: Union[int, Tuple[int, ...]] = -1
    temperature: float = 1.

    def __init__(
        self,
        model: PyTree,
        *,
        param_name: str = "weight",
        handler: Callable = None,
        axis: int = -1,
        minimum: float = 1e-3,
        smoothing: float = 0,
        temperature: float = 1.,
    ):
        self._image_map_impl = partial(
            paramsoftmax, temperature=temperature, axis=axis)
        self._preimage_map_impl = partial(
            paramlog, temperature=temperature, smoothing=smoothing)
        self.temperature = temperature
        super().__init__(
            model, param_name=param_name,
            image_bound=(minimum, float('inf')), # 1 - minimum),
            handler=handler
        )

    def preimage_map_impl(self, param: Tensor) -> Tensor:
        return self._preimage_map_impl(param)

    def image_map_impl(self, param: Tensor) -> Tensor:
        return self._image_map_impl(param)


class AmplitudeProbabilitySimplexParameter(
    PhaseAmplitudeMixin,
    ProbabilitySimplexParameter
):
    pass


class OrthogonalParameter(MappedParameter):
    def __init__(
        self,
        model: PyTree,
        *,
        param_name: str = "weight",
    ):
        super().__init__(
            model,
            param_name=param_name,
        )

    def preimage_map(self, param: Tensor) -> Tensor:
        return self.image_map(param)

    def image_map(self, param: Tensor) -> Tensor:
        return jnp.linalg.qr(param)[0]


class IsochoricParameter(DomainMappedParameter):
    volume: float = 1.
    max_condition: Optional[float] = None
    softmax_temp: Optional[float] = None

    def __init__(
        self,
        model: PyTree,
        *,
        param_name: str = "weight",
        volume: float = 1.,
        max_condition: Optional[float] = None,
        softmax_temp: Optional[float] = None,
        spd_threshold: float = 1e-3,
    ):
        self.volume = volume
        self.max_condition = max_condition
        self.softmax_temp = softmax_temp
        super().__init__(
            model, param_name=param_name,
            handler=ForcePositiveDefinite(spd_threshold=spd_threshold),
        )

    def preimage_map_impl(self, param: Tensor) -> Tensor:
        return self.image_map_impl(param)

    def image_map_impl(self, param: Tensor) -> Tensor:
        return isochor(
            param,
            volume=self.volume,
            max_condition=self.max_condition,
            softmax_temp=self.softmax_temp,
        )
