# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base initialisers for module parameters.
"""
import jax
import jax.numpy as jnp
import distrax
import equinox as eqx
from typing import Callable, Optional, Tuple
from ..functional.utils import PyTree, Tensor, Distribution


def from_distr_init(
    *,
    shape: Tuple[int],
    distr: Distribution,
    key: jax.random.PRNGKey,
) -> Tensor:
    """
    Sample a tensor from a specified distribution.

    Thin wrapper around ``distrax``.

    Parameters
    ----------
    shape : Tensor
        Shape of the tensor to populate or initialise from the specified
        distribution.
    distr : Distribution
        Distrax distribution object from which to sample values used to
        populate the tensor.
    """
    return distr.sample(seed=key, sample_shape=shape)


def constant_init(
    *,
    shape: Tuple[int],
    value: float = 0,
    key: Optional[jax.random.PRNGKey] = None
) -> Tensor:
    """
    Initialise a tensor to a constant value throughout. (The specified value
    doesn't actually have to be scalar as long as it is broadcastable to the
    tensor being initialised.)
    """
    return jnp.full(shape, value)


def identity_init(
    *,
    shape: Tuple[int],
    scale: float = 1,
    shift: float = 0,
    key: Optional[jax.random.PRNGKey] = None
) -> Tensor:
    """
    Initialise a tensor such that each of its slices is an identity matrix.
    Currently this sets each slice defined by the last two axes to identity.
    If there is a use case for other slices, it can be made more flexible in
    the future.
    """
    return jnp.tile(jnp.eye(shape[-1]) * scale + shift, (*shape[:-2], 1, 1))


class Initialiser(eqx.Module):
    def __call__(
        self,
        model: PyTree,
        *,
        param_name: str = "weight",
        key: jax.random.PRNGKey,
        **params,
    ):
        return self._init(
            shape=model.__getattribute__(param_name).shape,
            key=key,
            **params,
        )

    def _init(self, shape, key, **params):
        raise NotImplementedError

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        param_name: str = "weight",
        key: jax.random.PRNGKey,
        **params,
    ):
        init = cls()
        init = init(
            model=model, param_name=param_name, key=key, **params)
        return eqx.tree_at(
            lambda m: m.__getattribute__(param_name),
            model,
            replace=init
        )


class MappedInitialiser(Initialiser):
    mapper: Callable

    def __init__(
        self,
        mapper: Callable = None
    ):
        self.mapper = mapper

    def _init(self, shape, key):
        raise NotImplementedError

    @staticmethod
    def _init_impl(init, model, param_name, key, **params):
        model = eqx.tree_at(
            lambda m: m.__getattribute__(param_name),
            model,
            replace=init(
                model=model, param_name=param_name, key=key)
        )
        if init.mapper is None:
            return model
        return init.mapper.map(model=model, param_name=param_name, **params)

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Callable = None,
        param_name: str = "weight",
        key: jax.random.PRNGKey,
        **params,
    ):
        init = cls(mapper=mapper)
        return cls._init_impl(
            init=init,
            model=model,
            param_name=param_name,
            key=key,
            **params,
        )


class DistributionInitialiser(MappedInitialiser):
    """
    Parameter initialiser from a distribution.

    See :func:`from_distr_init` and :class:`MappedInitialiser` for usage
    details.
    """

    distribution : Distribution

    def __init__(
        self,
        distribution: Distribution,
        mapper: Callable = None
    ):
        self.distribution = distribution
        super().__init__(mapper=mapper)

    def _init(self, shape, key):
        return from_distr_init(
            shape=shape, distr=self.distribution, key=key)

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Callable = None,
        distribution: Distribution = None,
        param_name: str = "weight",
        key: jax.random.PRNGKey,
        **params,
    ):
        init = cls(mapper=mapper, distribution=distribution)
        return super()._init_impl(
            init=init, model=model, param_name=param_name, key=key, **params,
        )


class ConstantInitialiser(MappedInitialiser):
    value : float

    def __init__(
        self,
        value: float,
        mapper: Callable = None
    ):
        self.value = value
        super().__init__(mapper=mapper)

    def _init(self, shape, key):
        return constant_init(shape=shape, value=self.value, key=key)

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Callable = None,
        value: float = 0,
        param_name: str = "weight",
        key: jax.random.PRNGKey = None,
        **params,
    ):
        init = cls(mapper=mapper, value=value)
        return super()._init_impl(
            init=init, model=model, param_name=param_name, key=key, **params,
        )


class IdentityInitialiser(MappedInitialiser):
    scale : float
    shift : float

    def __init__(
        self,
        scale: float = 1,
        shift: float = 0,
        mapper: Callable = None
    ):
        self.scale = scale
        self.shift = shift
        super().__init__(mapper=mapper)

    def _init(self, shape, key):
        return identity_init(
            shape=shape, scale=self.scale, shift=self.shift, key=key)

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Callable = None,
        scale: float = 1,
        shift: float = 0,
        param_name: str = "weight",
        key: jax.random.PRNGKey = None,
        **params,
    ):
        init = cls(mapper=mapper, scale=scale, shift=shift)
        return super()._init_impl(
            init=init, model=model, param_name=param_name, key=key, **params,
        )


class DomainInitialiser:
    """
    Initialiser for a tensor whose values are the preimage of some function.

    For example, a layer can internally store a "preweight" that is passed
    through a logistic function to produce the actual weight seen by data in
    the forward pass. This constrains the actual weight to the interval (0, 1)
    and makes the unconstrained preweight the learnable parameter. We might
    often wish to initialise the actual weight from some distribution rather
    than initialising the preweight; this class provides a convenient way to
    do so.

    A ``DomainInitialiser`` is callable with a single required argument: a
    tensor to be initialised following the specified initialisation scheme.

    Parameters
    ----------
    init : callable
        A python callable that takes as its single required parameter the
        tensor that is to be initialised; the callable should, when called,
        initialise the tensor in place. Callables with additional arguments
        can be constrained using ``partial`` from ``functools`` or an
        appropriate lambda function. If no `init` is explicitly specified,
        ``DomainInitialiser`` defaults to a uniform initialisation in the
        interval (0, 1).
    domain : Domain object
        A representation of the function used to map between the learnable
        preweight and the weight "seen" by the data. It must have a
        ``preimage`` method that maps values in the weight domain to their
        preimage under the function: the corresponding values in the preweight
        domain. Examples are provided in
        :doc:`init.domain <hypercoil.init.domain>`.
        If no ``domain`` is
        explicitly specified, ``DomainInitialiser`` defaults to identity
        (preweight and weight are the same).
    """
    def __init__(self):
        raise NotImplementedError(
            'This deprecated functionality will be removed imminently')


class BaseInitialiser(DomainInitialiser):
    """
    Basic initialiser class. This class mostly exists to be subclassed.

    Parameters
    ----------
    init : callable
        A python callable that takes as its single required parameter the
        tensor that is to be initialised; the callable should, when called,
        initialise the tensor in place. Callables with additional arguments
        can be constrained using ``partial`` from ``functools`` or an
        appropriate lambda function. If no ``init`` is explicitly specified,
        ``BaseInitialiser`` defaults to a uniform initialisation in the
        interval (0, 1).
    """
    def __init__(self):
        raise NotImplementedError(
            'This deprecated functionality will be removed imminently')


class DistributionInitialiserDeprecated(DomainInitialiser):
    """
    Parameter initialiser from a distribution.

    See :func:`from_distr_init_` and :class:`DomainInitialiser` for argument
    details.
    """
    def __init__(self):
        raise NotImplementedError(
            'This deprecated functionality will be removed imminently')


class ConstantInitialiserDeprecated(DomainInitialiser):
    """
    Initialise a parameter to a constant value throughout.

    See :func:`constant_init_` and :class:`DomainInitialiser` for argument
    details.
    """
    def __init__(self):
        raise NotImplementedError(
            'This deprecated functionality will be removed imminently')


def from_distr_init_():
    raise NotImplementedError

def constant_init_():
    raise NotImplementedError

def identity_init_():
    raise NotImplementedError
