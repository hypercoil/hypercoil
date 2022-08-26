# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base initialisers for module parameters.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod
from typing import Optional, Tuple, Type
from .mapparam import MappedParameter
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
    """
    Initialiser base class.

    This class must be subclassed to be used. Subclasses must implement the
    ``_init`` method, which takes a desired output shape (``shape``) and a
    random number generator key (``key``) and returns a tensor of the
    requested shape.

    To use an initialiser, do not instantiate it directly, but instead use the
    ``init`` method of the module class that uses it. This will apply the
    initialiser to the model parameter specified by ``param_name``.
    """

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

    @abstractmethod
    def _init(
        self,
        shape: Tuple[int, ...],
        key: jax.random.PRNGKey,
        **params
    ) -> Tensor:
        raise NotImplementedError

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        param_name: str = "weight",
        key: jax.random.PRNGKey,
        **params,
    ) -> PyTree:
        init = cls()
        init = init(
            model=model, param_name=param_name, key=key, **params)
        return eqx.tree_at(
            lambda m: m.__getattribute__(param_name),
            model,
            replace=init
        )


class MappedInitialiser(Initialiser):
    """
    Parameter initialiser base class that also admits an optional
    :doc:`parameter map <api/hypercoil.init.mapparam>`.

    This is useful for combining initialisation and re-parameterisation into a
    single step. The supplied parameter map should not be an instance but a
    class. If no map is supplied, the initialiser is applied as normal.

    This class must be subclassed to be used. Subclasses must implement the
    ``_init`` method, which takes a desired output shape (``shape``) and a
    random number generator key (``key``) and returns a tensor of the
    requested shape.

    .. warning::
        To use an initialiser, do not instantiate it directly, but instead use
        the ``init`` method of the module class that uses it. This will apply
        the initialiser to the model parameter specified by ``param_name``.

    .. note::
        The initialiser is first used to initialise the requested parameter,
        and the mapping function is thereafter applied to the resulting
        tensor. If the initialisation produces out-of-domain values for the
        mapping function, the tensor that is ultimately instantiated might
        not reflect the specifications of the initialiser, as the mapping
        function will automatically apply any out-of-domain handlers.

    .. note::
        Any extra keyword arguments to the ``init`` method are passed to the
        mapping function when it is applied.
    """

    mapper: Optional[Type[MappedParameter]] = None

    def __init__(
        self,
        mapper: Optional[Type[MappedParameter]] = None
    ):
        self.mapper = mapper

    @abstractmethod
    def _init(self, shape, key):
        raise NotImplementedError

    @staticmethod
    def _init_impl(
        init: Initialiser,
        model: PyTree,
        param_name: str,
        key: Optional[jax.random.PRNGKey],
        **params
    ) -> PyTree:
        model = eqx.tree_at(
            lambda m: m.__getattribute__(param_name),
            model,
            replace=init(
                model=model, param_name=param_name, key=key)
        )
        if init.mapper is None:
            return model
        return init.mapper.map(model=model, param_name=param_name, **params)

    #TODO: This will be problematic if the mapper and the initialiser both
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
        param_name: str = "weight",
        key: jax.random.PRNGKey,
        **params,
    ) -> PyTree:
        """
        Initialise a parameter using the specified initialiser and mapper.

        Parameters
        ----------
        model : PyTree
            Model whose parameter to initialise.
        mapper : Optional[Type[MappedParameter]] (default: None)
            Mapping function to apply to the initialised parameter. This
            should be a subclass of
            :doc:`MappedParameter <api/hypercoil.init.mapparam.MappedParameter>`.
            If this is ``None``, the initialiser is applied as normal.
        param_name : str (default: "weight")
            Name of the parameter to initialise.
        key : jax.random.PRNGKey
            Pseudo-random number generator key to use for initialisation.
        **params : Any
            Extra keyword arguments; these are forwarded to the mapper when it
            is instantiated and applied.
        """
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
        mapper: Optional[Type[MappedParameter]] = None
    ):
        self.distribution = distribution
        super().__init__(mapper=mapper)

    def _init(
        self,
        shape: Tuple[int, ...],
        key: jax.random.PRNGKey,
    ) -> Tensor:
        return from_distr_init(
            shape=shape, distr=self.distribution, key=key)

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Optional[Type[MappedParameter]] = None,
        distribution: Distribution = None,
        param_name: str = "weight",
        key: jax.random.PRNGKey,
        **params,
    ) -> PyTree:
        init = cls(mapper=mapper, distribution=distribution)
        return super()._init_impl(
            init=init, model=model, param_name=param_name, key=key, **params,
        )


class ConstantInitialiser(MappedInitialiser):
    """
    Initialise a parameter to a constant value throughout.

    See :func:`constant_init` and :class:`MappedInitialiser` for argument
    details.
    """

    value : float

    def __init__(
        self,
        value: float,
        mapper: Optional[Type[MappedParameter]] = None
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
        param_name: str = "weight",
        key: jax.random.PRNGKey = None,
        **params,
    ) -> PyTree:
        init = cls(mapper=mapper, value=value)
        return super()._init_impl(
            init=init, model=model, param_name=param_name, key=key, **params,
        )


class IdentityInitialiser(MappedInitialiser):
    """
    Initialise a parameter such that all slices along the final two axes are
    identity matrices.

    See :func:`identity_init` and :class:`MappedInitialiser` for argument
    details.
    """

    scale : float
    shift : float

    def __init__(
        self,
        scale: float = 1,
        shift: float = 0,
        mapper: Optional[Type[MappedParameter]] = None
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
            shape=shape, scale=self.scale, shift=self.shift, key=key)

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Optional[Type[MappedParameter]] = None,
        scale: float = 1,
        shift: float = 0,
        param_name: str = "weight",
        key: jax.random.PRNGKey = None,
        **params,
    ) -> PyTree:
        init = cls(mapper=mapper, scale=scale, shift=shift)
        return super()._init_impl(
            init=init, model=model, param_name=param_name, key=key, **params,
        )


class DomainInitialiser:
    def __init__(self):
        raise NotImplementedError(
            'This deprecated functionality will be removed imminently')

class BaseInitialiser(DomainInitialiser):
    def __init__(self):
        raise NotImplementedError(
            'This deprecated functionality will be removed imminently')

class DistributionInitialiserDeprecated(DomainInitialiser):
    def __init__(self):
        raise NotImplementedError(
            'This deprecated functionality will be removed imminently')

class ConstantInitialiserDeprecated(DomainInitialiser):
    def __init__(self):
        raise NotImplementedError(
            'This deprecated functionality will be removed imminently')

def from_distr_init_():
    raise NotImplementedError

def constant_init_():
    raise NotImplementedError

def identity_init_():
    raise NotImplementedError
