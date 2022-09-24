# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Parameter mappers / mapped parameters for ``equinox`` modules.
Similar to PyTorch's ``torch.nn.utils.parametrize``.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from abc import abstractmethod
from functools import partial
from typing import Any, Callable, Literal, Optional, Tuple, Union
from ..engine.paramutil import (
    PyTree, Tensor, _to_jax_array, where_weight
)
from ..formula.nnops import retrieve_address
from ..functional.activation import isochor
from ..functional.matrix import spd
from ..functional.utils import (
    complex_decompose, complex_recompose
)


class MappedParameter(eqx.Module):
    """
    A transformed version of a parameter tensor.

    A ``MappedParameter`` wraps and replaces a standard parameter in a model.
    Subclasses can implement image (forward transformation) and preimage
    (often right inverse transformation) maps to transform the parameter
    tensor. At instantiation, the original parameter is mapped under the
    preimage map. It can thereafter be accessed in the ``original`` field of
    the ``MappedParameter``. Accessing the parameter instead accesses the
    tranformation of the original parameter under the image map.

    .. note::
        Rather than first instantiating a new ``MappedParameter`` and then
        creating an updated model, it is also possible to directly create an
        updated model that immediatelycontains the ``MappedParameter`` using
        the ``map`` class method.

    Parameters
    ----------
    model : PyTree
        The model to which the parameter belongs.
    where : Callable
        As in ``equinox.tree_at``: a function that takes a model (or generally
        a PyTree) and returns the parameter tensor to be mapped. For example:
        ``where = lambda mlp: mlp.layers[-1].linear.weight``. By default, the
        ``weight`` attribute of the model is retrieved.
    """

    original: Tensor

    def __init__(self, model: PyTree, *, where: Callable = where_weight):
        self.original = self.preimage_map(where(model))

    @abstractmethod
    def preimage_map(self, param: Tensor) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def image_map(self, param: Tensor) -> Tensor:
        raise NotImplementedError()

    def __jax_array__(self):
        return self.image_map(_to_jax_array(self.original))

    @classmethod
    def map(
        cls,
        model: PyTree,
        *pparams,
        where: Union[str, Callable] = "weight",
        **params
    ):
        """
        Create an updated version of a model that contains the mapped
        parameter.

        Parameters
        ----------
        model : PyTree
            The model to which the parameter belongs.
        where : str or Callable (default: `"weight"`)
            The address of the parameter to be mapped, specified as a string
            or a function. For example: ``where = "weight"`` or
            ``where = lambda mlp: mlp.layers[-1].linear.weight``.

        Returns
        -------
        model : PyTree
            The updated model containing the mapped parameter.
        """
        #TODO: We're inefficiently making a lot of repeated calls to
        #      ``retrieve_address`` here. We might be able to do this more
        #      efficiently, but this is low-priority as each call usually has
        #      very little overhead.
        parameters = retrieve_address(model, where)
        mapped = ()
        for i, _ in enumerate(parameters):
            where_i = lambda model: retrieve_address(model, where)[i]
            mapped += (cls(
                model=model,
                *pparams,
                where=where_i,
                **params),)
        return eqx.tree_at(
            lambda m: retrieve_address(m, where),
            model,
            replace=mapped,
        )


class OutOfDomainHandler(eqx.Module):
    """
    Generic object for evaluating whether values in a parameter tensor are
    within a domain and applying a transformation to those values that are
    outside the domain.
    """
    def test(self, x: Tensor, bound: Tuple[float, float]) -> Tensor:
        """
        Evaluate whether each entry in a tensor falls within bounds.

        Parameters
        ----------
        x : Tensor
            Tensor whose entries should be evaluated.
        bound : (float min, float max)
            Minimum and maximum permitted values for the test to return True.

        Returns
        -------
        result : Tensor
            Boolean-valued tensor indicating whether each entry of the input
            is in the prescribed bounds.
        """
        return jnp.logical_and(
            x <= bound[-1],
            x >= bound[0]
        )


class Clip(OutOfDomainHandler):
    """
    Handle out-of-domain values by clipping them to the closest allowed point.
    """
    def apply(self, x: Tensor, bound: Tuple[float, float]) -> Tensor:
        """
        Clip values in the specified tensor to the specified bounds.

        Parameters
        ----------
        x : Tensor
            Tensor whose out-of-domain entries are to be clipped.
        bound : (float min, float max)
            Minimum and maximum permitted values; values greater than the
            maximum will be clipped to the maximum and values less than the
            minimum will be clipped to the minimum.

        Returns
        -------
        out : Tensor
            Copy of the input tensor with out-of-domain entries clipped.
        """
        x = jax.lax.stop_gradient(x)
        return jnp.clip(x, bound[0], bound[-1])


class Renormalise(OutOfDomainHandler):
    """
    Normalise entries in the specified tensor so that all are in an interval
    specified by bounds.

    Note that this handler is in general much more radical than clipping. In
    particular, if any entry at all is out of bounds, nearly all entries in
    the tensor will be edited, and a single extreme outlier can destroy the
    variance in the dataset. If you are considering using this because most of
    the data you expect to see will be outside of the prescribed domain,
    consider using a different domain first (for instance, using the ``scale``
    parameter to accommodate a larger feasible interval).

    The normalisation procedure works by mapping the original range of
    observations, [``obs_min``, ``obs_max``], to
    [max(``obs_min``, ``lbound``), min(``obs_max``, ``ubound``)] while
    preserving relative distances between observations.
    """
    def apply(
        self,
        x: Tensor,
        bound: Tuple[float, float],
        axis: Optional[Tuple[int, ...]] = None
    ) -> Tensor:
        """
        Re-scale all values in the specified tensor to fall in the specified
        interval.

        Parameters
        ----------
        x : Tensor
            Tensor whose out-of-domain entries are to be normalised.
        bound : (float min, float max)
            Minimum and maximum permitted values.

        Returns
        -------
        out : Tensor
            Copy of the input tensor with all entries normalised.
        """
        x = jax.lax.stop_gradient(x)
        upper = x.max(axis)
        lower = x.min(axis)
        unew = jnp.minimum(bound[-1], upper)
        lnew = jnp.maximum(bound[0], lower)
        out = x - x.mean(axis)
        out = out / ((upper - lower) / (unew - lnew))
        return out + lnew - out.min(axis)


class ForcePositiveDefinite(OutOfDomainHandler):
    """
    Handle non-positive definite slices of a parameter tensor by forcing them
    to be positive definite.
    """
    def apply(
        self,
        x: Tensor,
        bound: Tuple[float, float],
    ) -> Tensor:
        """
        Force all non-positive definite slices of the specified tensor to be
        positive definite.

        Parameters
        ----------
        x : Tensor
            Tensor whose out-of-domain slices are to be forced to be positive
            definite.
        bound : (float min, float max)
            Only the minimal bound is used. It specifies the minimum
            permissible eigenvalue of each matrix slice.
        """
        eps = bound[0]
        x = jax.lax.stop_gradient(x)
        return spd(x, eps=eps)

    def test(self, *pparams, **params):
        """
        Not implemented.
        """
        raise NotImplementedError()


class DomainMappedParameter(MappedParameter):
    """
    A parameter that is mapped between different domains.

    This extends :class:`MappedParameter` in that it admits for the forward
    (image) and backward (preimage) maps (i) a set of bounds on the domain,
    and (ii) a mechanism for handling out-of-domain values.

    Parameters
    ----------
    model : PyTree
        The model to which the parameter belongs.
    where : Callable
        As in ``equinox.tree_at``: a function that takes a model (or generally
        a PyTree) and returns the parameter tensor to be mapped. For example:
        ``where = lambda mlp: mlp.layers[-1].linear.weight``. By default, the
        ``weight`` attribute of the model is retrieved.
    image_bound : Tuple[float, float]
        Minimum and maximum prescribed values for the image map. Note that
        these are not necessarily the same as the minimum and maximum (or
        infimum and supremum) attainable by the map itself. For example, it
        might be desirable to further truncate some maps to a range where the
        magnitude of the gradient is non-negligible. However, in general, the
        prescribed bounds should not admit any values outside the domain of
        the map.
    preimage_bound : Tuple[float, float]
        Minimum and maximum prescribed values for the preimage map.
    handler : ``OutOfDomainHandler`` object (default: :class:`Clip`)
        The handler to use for imputing out-of-domain values.
    """

    original: Tensor
    image_bound: Any = None
    preimage_bound: Any = None
    handler: Any = None

    def __init__(
        self,
        model: PyTree,
        *,
        where: Callable = where_weight,
        image_bound: Any = None,
        preimage_bound: Any = None,
        handler: Callable = None
    ):
        self.handler = handler or Clip()
        self.image_bound = image_bound or (-float('inf'), float('inf'))
        self.preimage_bound = preimage_bound or (-float('inf'), float('inf'))
        super(DomainMappedParameter, self).__init__(model=model, where=where)

    def preimage_map(self, param: Tensor) -> Tensor:
        """
        Map a tensor to its preimage under the transformation. Any values
        outside the prescribed bounds are handled as appropriate.
        """
        x = self.handler.apply(param, self.image_bound)
        i = self.preimage_map_impl(x)
        i = self.handler.apply(i, self.preimage_bound)
        return i

    def image_map(self, param: Tensor) -> Tensor:
        """
        Map a tensor to its image under the transformation.
        """
        return self.image_map_impl(param)

    @abstractmethod
    def preimage_map_impl(self, param: Tensor) -> Tensor:
        raise NotImplementedError()

    @abstractmethod
    def image_map_impl(self, param: Tensor) -> Tensor:
        raise NotImplementedError()

    def test(self, param: Tensor):
        """
        Evaluate whether each entry in a tensor falls within the prescribed
        bounds of the image map.
        """
        return self.handler.test(param, self.image_bound)

    def handle_ood(self, param: Tensor) -> Tensor:
        """
        Apply an out-of-domain handler to ensure that all tensor entries are
        within bounds.
        """
        return self.handler.apply(param, self.image_bound)


class AffineDomainMappedParameter(DomainMappedParameter):
    """
    A parameter that is mapped between different domains.

    This extends :class:`DomainMappedParameter` in that it also accepts
    arguments specifying a location and scale for the transformation.

    Parameters
    ----------
    model : PyTree
        The model to which the parameter belongs.
    where : Callable
        As in ``equinox.tree_at``: a function that takes a model (or generally
        a PyTree) and returns the parameter tensor to be mapped. For example:
        ``where = lambda mlp: mlp.layers[-1].linear.weight``. By default, the
        ``weight`` attribute of the model is retrieved.
    image_bound : Tuple[float, float]
        Minimum and maximum prescribed values for the image map. See the
        documentation for :class:`DomainMappedParameter` for more details.
    preimage_bound : Tuple[float, float]
        Minimum and maximum prescribed values for the preimage map.
    handler : ``OutOfDomainHandler`` object (default: :class:`Clip`)
        The handler to use for imputing out-of-domain values.
    loc : Tensor
        Location of a scalar affine transformation that is composed with the
        map.
    scale : Tensor
        Scale of a scalar affine transformation that is composed with the map.
    """

    original: Tensor
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
        where: Callable = where_weight,
        image_bound: Any = None,
        preimage_bound: Any = None,
        handler: Callable = None
    ):
        self.loc = loc
        self.scale = scale
        super(AffineDomainMappedParameter, self).__init__(
            model=model,
            where=where,
            image_bound=image_bound,
            preimage_bound=preimage_bound,
            handler=handler
        )

    def preimage_map(self, param: Tensor) -> Tensor:
        """
        Map a tensor to its preimage under the transformation. Any values
        outside the prescribed bounds are handled as appropriate.
        """
        x = self.handler.apply(param, self.image_bound)
        i = self.preimage_map_impl((x - self.loc) / self.scale)
        i = self.handler.apply(i, self.preimage_bound)
        return i

    def image_map(self, param: Tensor) -> Tensor:
        """
        Map a tensor to its image under the transformation.
        """
        return self.scale * self.image_map_impl(param) + self.loc


class PhaseAmplitudeMixin:
    """
    A mixin class for mapped parameters.

    Inheriting this mixin class changes any mapped parameter such that its
    mapping is applied specifically to the amplitude of a complex-valued
    parameter. The phase is left unchanged.
    """
    def preimage_map(self, param: Tensor) -> Tensor:
        """
        Map the amplitude of a complex-valued tensor to its preimage under the
        transformation. Any values outside the transformation's range are
        first handled.
        """
        ampl, phase = complex_decompose(param)
        ampl = super().preimage_map(ampl)
        return complex_recompose(ampl, phase)

    def image_map(self, param: Tensor) -> Tensor:
        """
        Map the amplitude of a complex-valued a tensor to its image under the
        transformation.
        """
        ampl, phase = complex_decompose(param)
        ampl = super().image_map(ampl)
        return complex_recompose(ampl, phase)


class IdentityMappedParameter(MappedParameter):
    """Identity parameter mapper."""

    def preimage_map(self, param: Tensor) -> Tensor:
        return param

    def image_map(self, param: Tensor) -> Tensor:
        return param


class AffineMappedParameter(AffineDomainMappedParameter):
    """Affine parameter mapper."""

    def preimage_map_impl(self, param: Tensor) -> Tensor:
        return param

    def image_map_impl(self, param: Tensor) -> Tensor:
        return param


class TanhMappedParameter(AffineDomainMappedParameter):
    """
    Parameter mapped through a hyperbolic tangent function. The tensor values
    are thus constrained between some finite scale value and its negation.

    Parameters
    ----------
    model : PyTree
        The model to which the parameter belongs.
    where : Callable
        As in ``equinox.tree_at``: a function that takes a model (or generally
        a PyTree) and returns the parameter tensor to be mapped. For example:
        ``where = lambda mlp: mlp.layers[-1].linear.weight``. By default, the
        ``weight`` attribute of the model is retrieved.
    preimage_bound : (float min, float max) (default -3, 3)
        Minimum and maximum prescribed values for the preimage map. Note that
        these are not necessarily the same as the minimum and maximum (or
        infimum and supremum) attainable by the map itself. For example, it
        might be desirable to further truncate values to a range where the
        magnitude of the gradient is non-negligible.
    handler : ``OutOfDomainHandler`` object (default :class:`Clip`)
        Object specifying a method for handling out-of-domain entries.
    scale : float (default 1)
        Maximum/minimum value attained by the hyperbolic tangent map.
    """
    def __init__(
        self,
        model: PyTree,
        *,
        where: Callable = where_weight,
        preimage_bound: Tuple[float, float] = (-3., 3.),
        handler: Callable = None,
        scale: float = 1.,
    ):
        super().__init__(
            model,
            where=where,
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
    """
    Complex-valued parameter whose amplitude is mapped through a hyperbolic
    tangent function. The amplitude is thus constrained between some finite
    scale value and its negation.

    Parameters
    ----------
    model : PyTree
        The model to which the parameter belongs.
    where : Callable
        As in ``equinox.tree_at``: a function that takes a model (or generally
        a PyTree) and returns the parameter tensor to be mapped. For example:
        ``where = lambda mlp: mlp.layers[-1].linear.weight``. By default, the
        ``weight`` attribute of the model is retrieved.
    preimage_bound : (float min, float max) (default -3, 3)
        Minimum and maximum prescribed values for the preimage map. Note that
        these are not necessarily the same as the minimum and maximum (or
        infimum and supremum) attainable by the map itself. For example, it
        might be desirable to further truncate values to a range where the
        magnitude of the gradient is non-negligible.
    handler : ``OutOfDomainHandler`` object (default :class:`Clip`)
        Object specifying a method for handling out-of-domain entries.
    scale : float (default 1)
        Maximum/minimum value attained by the hyperbolic tangent map.
    """


class MappedLogits(AffineDomainMappedParameter):
    """
    Parameter mapped through a logistic function. The tensor values are thus
    constrained between 0 and 1, or more generally between 2 arbitrary finite
    real numbers.

    Parameters
    ----------
    model : PyTree
        The model to which the parameter belongs.
    where : Callable
        As in ``equinox.tree_at``: a function that takes a model (or generally
        a PyTree) and returns the parameter tensor to be mapped. For example:
        ``where = lambda mlp: mlp.layers[-1].linear.weight``. By default, the
        ``weight`` attribute of the model is retrieved.
    preimage_bound : (float min, float max) (default -3, 3)
        Minimum and maximum prescribed values for the preimage map. Note that
        these are not necessarily the same as the minimum and maximum (or
        infimum and supremum) attainable by the map itself. For example, it
        might be desirable to further truncate values to a range where the
        magnitude of the gradient is non-negligible.
    handler : ``OutOfDomainHandler`` object (default :class:`Clip`)
        Object specifying a method for handling out-of-domain entries.
    loc : float (default None)
        Location parameter for the logistic map. Zero is mapped to this value
        under the logistic map.
    scale : float (default 1)
        Size of the interval mapped onto by the logistic map.
    """
    def __init__(
        self,
        model: PyTree,
        *,
        where: Callable = where_weight,
        preimage_bound: Tuple[float, float] = (-4.5, 4.5),
        handler: Callable = None,
        loc: Optional[float] = None,
        scale: float = 1.,
    ):
        if loc is None: loc = (scale / 2)
        shift = loc - scale / 2
        super().__init__(
            model,
            where=where,
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
    """
    Parameter whose constituent vectors are projected onto a sphere of some
    fixed norm.

    Note that this will only work for proper convex norms. (It will obviously
    not work for the L0 norm, for example.)

    .. warning::
        The normalisation procedure is a simple division operation. Thus, it
        will not work for many matrix norms.

    Parameters
    ----------
    model : PyTree
        The model to which the parameter belongs.
    where : Callable
        As in ``equinox.tree_at``: a function that takes a model (or generally
        a PyTree) and returns the parameter tensor to be mapped. For example:
        ``where = lambda mlp: mlp.layers[-1].linear.weight``. By default, the
        ``weight`` attribute of the model is retrieved.
    handler : ``OutOfDomainHandler`` object (default :class:`Clip`)
        Object specifying a method for handling out-of-domain entries.
    loc : float or tensor
        Location of the centre of the norm sphere. By default, the sphere is
        centred at the origin (``loc=0``).
    scale : float (default 1)
        Scale of the norm sphere. By default, the unit norm sphere
        (``scale=1``) is used.
    norm : int, str, or tensor
        Norm order argument passed to ``jnp.linalg.norm``. If this is
        a tensor of symmetric positive semidefinite matrices, then a
        Mahalanobis-like ellipse norm is computed using those matrices. (No
        matrix inversion is performed.) By default, the Euclidean norm is
        used.
    axis : int or tuple(int)
        Axis or axis tuple over which the norm is computed. Every slice along
        the specified axis or axis tuple of the mapped tensor is rescaled so
        that its norm is equal to ``scale`` in the specified ``norm``.
    """

    normalise_fn: Callable
    order: Tensor = 2
    axis: Union[int, Tuple[int, ...]] = -1

    def __init__(
        self,
        model: PyTree,
        *,
        where: Callable = where_weight,
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
            model, where=where,
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
    """
    Parameter whose constituent slices are projected onto the probability
    simplex.

    The forward function is a softmax. Note that the softmax function does
    not have a unique inverse; here we use the elementwise natural logarithm
    as an 'inverse'. For a relatively well-behaved map, pair this with
    :doc:`Dirichlet initialisation <init.dirichlet.DirichletInit>`.

    Parameters
    ----------
    model : PyTree
        The model to which the parameter belongs.
    where : Callable
        As in ``equinox.tree_at``: a function that takes a model (or generally
        a PyTree) and returns the parameter tensor to be mapped. For example:
        ``where = lambda mlp: mlp.layers[-1].linear.weight``. By default, the
        ``weight`` attribute of the model is retrieved.
    handler : ``OutOfDomainHandler`` object (default :class:`Clip`)
        Object specifying a method for handling out-of-domain entries.
    axis : int (default -1)
        Axis of tensors in the domain along which 1D slices are mapped to the
        probability simplex.
    minimum : nonnegative float (default 1e-3)
        Lower prescribed bound on inputs to the elementwise natural logarithm.
    smoothing : nonnegative float (default 0)
        For use when configuring the ``original`` parameter. Increasing the
        smoothing will result in a higher-entropy / more equiprobable image.
    temperature : nonnegative float or ``'auto'`` (default 1)
        Softmax temperature.
    """

    _image_map_impl: Callable
    _preimage_map_impl: Callable
    axis: Union[int, Tuple[int, ...]] = -1
    temperature: float = 1.

    def __init__(
        self,
        model: PyTree,
        *,
        where: Callable = where_weight,
        handler: Callable = None,
        axis: int = -1,
        minimum: float = 1e-3,
        smoothing: float = 0,
        temperature: Union[float, Literal['auto']] = 1.,
    ):
        if temperature == 'auto':
            temperature = jnp.sqrt(where(model).shape[axis])
        self._image_map_impl = partial(
            paramsoftmax, temperature=temperature, axis=axis)
        self._preimage_map_impl = partial(
            paramlog, temperature=temperature, smoothing=smoothing)
        self.temperature = temperature
        super().__init__(
            model, where=where,
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
    """
    Complex-valued parameter whose amplitudes are projected onto the
    probability simplex.

    The forward function is a softmax applied to the amplitudes. Note that
    the softmax function does not have a unique inverse; here we use the
    elementwise natural logarithm as an 'inverse'.

    Parameters
    ----------
    model : PyTree
        The model to which the parameter belongs.
    where : Callable
        As in ``equinox.tree_at``: a function that takes a model (or generally
        a PyTree) and returns the parameter tensor to be mapped. For example:
        ``where = lambda mlp: mlp.layers[-1].linear.weight``. By default, the
        ``weight`` attribute of the model is retrieved.
    handler : ``OutOfDomainHandler`` object (default :class:`Clip`)
        Object specifying a method for handling out-of-domain entries.
    axis : int (default -1)
        Axis of tensors in the domain along which 1D slices are mapped to the
        probability simplex.
    minimum : nonnegative float (default 1e-3)
        Lower prescribed bound on inputs to the elementwise natural logarithm.
    smoothing : nonnegative float (default 0)
        For use when configuring the ``original`` parameter. Increasing the
        smoothing will result in a higher-entropy / more equiprobable image.
    temperature : nonnegative float (default 1)
        Softmax temperature.
    """


class OrthogonalParameter(MappedParameter):
    """
    Parameter whose constituent slices are orthogonal vectors.

    Currently, this is implemented in a crude manner using a QR
    decomposition.

    Parameters
    ----------
    model : PyTree
        The model to which the parameter belongs.
    where : Callable
        As in ``equinox.tree_at``: a function that takes a model (or generally
        a PyTree) and returns the parameter tensor to be mapped. For example:
        ``where = lambda mlp: mlp.layers[-1].linear.weight``. By default, the
        ``weight`` attribute of the model is retrieved.
    """
    def __init__(
        self,
        model: PyTree,
        *,
        where: Callable = where_weight,
    ):
        super().__init__(
            model,
            where=where,
        )

    def preimage_map(self, param: Tensor) -> Tensor:
        return self.image_map(param)

    def image_map(self, param: Tensor) -> Tensor:
        return jnp.linalg.qr(param)[0]


class IsochoricParameter(DomainMappedParameter):
    """
    Parameter representing an ellipsoid of fixed volume.

    .. note::
        This parameter mapping requires the original parameter to be square
        and symmetric along its last 2 axes. The parameter is further forced
        to be positive definite.

    Parameters
    ----------
    model : PyTree
        The model to which the parameter belongs.
    where : Callable
        As in ``equinox.tree_at``: a function that takes a model (or generally
        a PyTree) and returns the parameter tensor to be mapped. For example:
        ``where = lambda mlp: mlp.layers[-1].linear.weight``. By default, the
        ``weight`` attribute of the model is retrieved.
    volume : nonnegative float (default 1)
        Parameter volume. The determinant of the parameter is set to this
        value.
    max_condition : float :math:`\in [1, \infty)` or None (default None)
        Maximum permissible condition number of the parameter. This can be
        used to constrain the eccentricity of isochoric ellipsoids. To enforce
        this maximum, the eigenvalues of the original parameter are replaced
        with a convex combination of the original eigenvalues and a vector of
        ones such that the largest eigenvalue is no more than
        ``max_condition`` times the smallest eigenvalue. Note that a
        ``max_condition`` of 1 will always return (a potentially isotropically
        scaled) identity.
    softmax_temp : float or None (default None)
        If this is provided, then the eigenvalues of the original parameter
        are passed through a softmax with the specified temperature before any
        other processing.
    """

    volume: float = 1.
    max_condition: Optional[float] = None
    softmax_temp: Optional[float] = None

    def __init__(
        self,
        model: PyTree,
        *,
        where: Callable = where_weight,
        volume: float = 1.,
        max_condition: Optional[float] = None,
        softmax_temp: Optional[float] = None,
        spd_threshold: float = 1e-3,
    ):
        self.volume = volume
        self.max_condition = max_condition
        self.softmax_temp = softmax_temp
        super().__init__(
            model, where=where,
            handler=ForcePositiveDefinite(),
            preimage_bound=(spd_threshold, float('inf')),
            image_bound=(spd_threshold, float('inf')),
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
