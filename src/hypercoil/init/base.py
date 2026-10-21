# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base initialisers for module parameters.
"""
from __future__ import annotations
from abc import abstractmethod
from functools import partial
from typing import Callable, Optional, Tuple, Type, Union

import jax
import jax.numpy as jnp
import equinox as eqx

from ..engine.paramutil import Distribution, PyTree, Tensor
from ..formula.nnops import retrieve_address
from .mapparam import MappedParameter


def from_distr_init(
    *,
    shape: Tuple[int],
    distr: Distribution,
    key: jax.random.PRNGKey,
) -> Tensor:
    """
    Sample a tensor from a specified distribution.

    Thin wrapper around ``numpyro``.

    Parameters
    ----------
    shape
        Shape of the tensor to populate or initialise from the specified
        distribution.
    distr
        Numpyro distribution object from which to sample values used to
        populate the tensor.
    """
    return distr.sample(key=key, sample_shape=shape)


def constant_init(
    *,
    shape: Tuple[int],
    value: float = 0,
    key: Optional[jax.random.PRNGKey] = None,
) -> Tensor:
    """
    Initialise a tensor to a constant value.

    The value need not be scalar; it is broadcast to the tensor shape.
    """
    return jnp.full(shape, value)


def identity_init(
    *,
    shape: Tuple[int],
    scale: float = 1,
    shift: float = 0,
    key: Optional[jax.random.PRNGKey] = None,
) -> Tensor:
    """
    Initialise a tensor with identity matrices over the last two axes.
    """
    return jnp.tile(jnp.eye(shape[-1]) * scale + shift, (*shape[:-2], 1, 1))


class Initialiser(eqx.Module):
    """
    Initialiser base class.

    This class must be subclassed to be used. Subclasses must implement the
    ``_init`` method, which takes a desired output shape (``shape``) and a
    random number generator key (``key``) and returns a tensor of the
    requested shape.

    To use an initialiser, do not instantiate it directly, but instead use the
    ``init`` method of the module class that uses it. This will apply the
    initialiser to the model parameter specified by ``where``.
    """

    def __call__(
        self,
        model: PyTree,
        *,
        where: Union[str, Callable] = 'weight',
        key: jax.random.PRNGKey,
        **params,
    ):
        parameters = retrieve_address(model, where=where)
        if key is not None:
            keys = jax.random.split(key, len(parameters))
        else:
            keys = (None,) * len(parameters)
        return tuple(
            self._init(
                shape=parameter.shape,
                key=key,
                **params,
            )
            for key, parameter in zip(keys, parameters)
        )

    @abstractmethod
    def _init(
        self,
        shape: Tuple[int, ...],
        key: jax.random.PRNGKey,
        **params,
    ) -> Tensor:
        raise NotImplementedError

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        where: Union[str, Callable] = 'weight',
        key: jax.random.PRNGKey,
        **params,
    ) -> PyTree:
        init = cls()
        init = init(model=model, where=where, key=key, **params)
        return eqx.tree_at(
            partial(retrieve_address, where=where),
            model,
            replace=init,
        )


class MappedInitialiser(Initialiser):
    """
    Initialiser base class with an optional parameter map.

    The :doc:`parameter map <api/hypercoil.init.mapparam>` combines
    initialisation and re-parameterisation in one step. The map is given as
    a class, not an instance. Without a map, the initialiser is applied as
    normal.

    A subclass implements ``_init``, which takes an output shape (``shape``)
    and a random number generator key (``key``) and returns a tensor of that
    shape.

    .. warning::
        Do not call an instance directly. The classmethod ``init`` applies
        the initialiser to the model parameter at ``where``.

    .. note::
        The initialiser fills the parameter first, and the map is then
        applied to the result. Because the map's out-of-domain handler
        transforms any out-of-domain values, the instantiated tensor might
        not match the initialiser's specification.

    .. note::
        Extra keyword arguments to ``init`` are passed to the map.
    """

    mapper: Optional[Type[MappedParameter]] = None

    def __init__(
        self,
        mapper: Optional[Type[MappedParameter]] = None,
    ):
        self.mapper = mapper

    @abstractmethod
    def _init(self, shape: Tuple[int, ...], key: jax.random.PRNGKey) -> Tensor:
        ...

    @staticmethod
    def _init_impl(
        init: Initialiser,
        model: PyTree,
        where: Union[str, Callable],
        key: Optional[jax.random.PRNGKey],
        **params,
    ) -> PyTree:
        model = eqx.tree_at(
            partial(retrieve_address, where=where),
            model,
            replace=init(model=model, where=where, key=key),
        )
        if init.mapper is None:
            return model
        return init.mapper.map(model=model, where=where, **params)

    # TODO: This will be problematic if the mapper and the initialiser both
    #      use the same parameter name. In this case the initialiser will
    #      consume the parameter and it will fail to reach the mapper.
    #
    #      This is a major problem and should be addressed with highest
    #      priority.
    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Optional[Type[MappedParameter]] = None,
        where: Union[str, Callable] = 'weight',
        key: jax.random.PRNGKey,
        **params,
    ) -> PyTree:
        """
        Initialise a parameter using the specified initialiser and mapper.

        Parameters
        ----------
        model
            Model whose parameter to initialise.
        mapper
            Mapping function to apply to the initialised parameter. This
            should be a subclass of
            :doc:`MappedParameter <api/hypercoil.init.mapparam.MappedParameter>`.
            If this is ``None``, the initialiser is applied as normal.
        where
            Address of the parameter to initialise. This can be a string
            address or a callable that takes a model and returns a
            parameter.
        key
            Pseudo-random number generator key to use for initialisation.
        **params : Any
            Extra keyword arguments; these are forwarded to the mapper when it
            is instantiated and applied.
        """
        init = cls(mapper=mapper)
        return cls._init_impl(
            init=init,
            model=model,
            where=where,
            key=key,
            **params,
        )


class DistributionInitialiser(MappedInitialiser):
    """
    Parameter initialiser from a distribution.

    See :func:`from_distr_init` and :class:`MappedInitialiser` for usage
    details.
    """

    distribution: Distribution

    def __init__(
        self,
        distribution: Distribution,
        mapper: Optional[Type[MappedParameter]] = None,
    ):
        self.distribution = distribution
        super().__init__(mapper=mapper)

    def _init(
        self,
        shape: Tuple[int, ...],
        key: jax.random.PRNGKey,
    ) -> Tensor:
        return from_distr_init(
            shape=shape,
            distr=self.distribution,
            key=key,
        )

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Optional[Type[MappedParameter]] = None,
        distribution: Distribution = None,
        where: Union[str, Callable] = 'weight',
        key: jax.random.PRNGKey,
        **params,
    ) -> PyTree:
        init = cls(mapper=mapper, distribution=distribution)
        return super()._init_impl(
            init=init,
            model=model,
            where=where,
            key=key,
            **params,
        )


class ConstantInitialiser(MappedInitialiser):
    """
    Initialise a parameter to a constant value throughout.

    See :func:`constant_init` and :class:`MappedInitialiser` for argument
    details.
    """

    value: float

    def __init__(
        self,
        value: float,
        mapper: Optional[Type[MappedParameter]] = None,
    ):
        self.value = value
        super().__init__(mapper=mapper)

    def _init(
        self,
        shape: Tuple[int, ...],
        key: jax.random.PRNGKey,
    ) -> Tensor:
        return constant_init(shape=shape, value=self.value, key=key)

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Optional[Type[MappedParameter]] = None,
        value: float = 0,
        where: Union[str, Callable] = 'weight',
        key: jax.random.PRNGKey = None,
        **params,
    ) -> PyTree:
        init = cls(mapper=mapper, value=value)
        return super()._init_impl(
            init=init,
            model=model,
            where=where,
            key=key,
            **params,
        )


class IdentityInitialiser(MappedInitialiser):
    """
    Initialiser that sets each slice over the last two axes to identity.

    See :func:`identity_init` and :class:`MappedInitialiser` for argument
    details.
    """

    scale: float
    shift: float

    def __init__(
        self,
        scale: float = 1,
        shift: float = 0,
        mapper: Optional[Type[MappedParameter]] = None,
    ):
        self.scale = scale
        self.shift = shift
        super().__init__(mapper=mapper)

    def _init(
        self,
        shape: Tuple[int, ...],
        key: jax.random.PRNGKey,
    ) -> Tensor:
        return identity_init(
            shape=shape, scale=self.scale, shift=self.shift, key=key
        )

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Optional[Type[MappedParameter]] = None,
        scale: float = 1,
        shift: float = 0,
        where: Union[str, Callable] = 'weight',
        key: jax.random.PRNGKey = None,
        **params,
    ) -> PyTree:
        init = cls(mapper=mapper, scale=scale, shift=shift)
        return super()._init_impl(
            init=init,
            model=model,
            where=where,
            key=key,
            **params,
        )
