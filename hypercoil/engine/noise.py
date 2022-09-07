# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Noise
~~~~~
Additive and multiplicative noise sources.
"""
import math
import jax
import jax.numpy as jnp
import distrax
import equinox as eqx
from abc import abstractmethod
from typing import Any, Callable, Optional, Sequence, Tuple
from .paramutil import (
    Distribution, PyTree, Tensor, _to_jax_array, where_weight
)
from .axisutil import argsort, standard_axis_number, axis_complement
from ..formula.nnops import retrieve_parameter


def _symlog(input):
    """
    Symmetric logarithmic transform. The bare functionality here is
    necessary so we don't import from ``functional``. In practice, the
    ``symlog`` function in ``functional`` is more robust and flexible, and
    should be favoured where possible.
    """
    L, Q = jnp.linalg.eigh((input + input.swapaxes(-2, -1)) / 2)
    Lmap = map(L)
    Lmap = jnp.where(jnp.isnan(Lmap), 0, Lmap)
    output = (Q @ (Lmap[..., None] * Q.swapaxes(-1, -2)))
    return (output + output.swapaxes(-2, -1)) / 2


#TODO: eqx.filter isn't compatible with distrax.Distribution objects.
#      To get around this, we currently set distributions as static fields.
#      This is a hack, but it should work for most of our needs as long as
#      the distribution is the same instance across epoch updates.
#      In the long run, we will want to figure out the cause for this
#      incompatibility.
#
#      Opened an issue on distrax:
#      https://github.com/deepmind/distrax/issues/193


def document_stochastic_transforms(func):
    base_warning = """
    .. warning::
        When training models with stochastic features, you must apply the
        :func:`refresh` function to the parent model to ensure that each
        transform has a fresh random number generator key. Otherwise, any
        stochastic transforms will produce the same stale samples across
        epochs."""
    base_param_spec = """
    inference : bool, optional (default: ``False``)
        Indicates that the transform (or its parent model) is in inference
        mode. If False, the transform will inject noise into any inputs it
        receives in its ``__call__`` method.
    key : jax.random.PRNGKey
        The random number generator key to use when sampling from the noise
        distribution. To refresh stale keys so that they produce new noise,
        you must use the :func:`refresh` function on the transform or its
        parent model.
    refresh_code : Any, optional (default: ``0``)
        The code to use when filtering stochastic sources. The default code is
        0. In nearly all cases, it is not necessary to change the default
        value."""
    axial_param_spec = """
    sample_axes : Tuple[int, ...] (default: None)
        By default, a sample is drawn from the distribution independently for
        each element of the input. If ``sample_axes`` is a value other than
        None, then a sample is drawn from the distribution only along the
        specified axes; this sample is then broadcast across the remaining
        axes of the input."""
    iid_scalar_param_spec = """
    distribution : ``distrax.Distribution``
        The distribution to sample from. The ``event_shape`` of this
        distribution should correspond to a scalar."""
    iid_tensor_param_spec = """
    distribution : ``distrax.Distribution``
        The distribution to sample from. The ``event_shape`` of this
        distribution should match the shape of the input along a set of axes,
        which must be specified in the ``event_axes`` argument.
    event_axes : Tuple[int, ...]
        Specifies the axes of the input tensor that correspond to the event
        shape of the distribution. Note that event axes are automatically
        excluded from the sample axes."""
    multiplicative_mean_correction = """
    If the mean of the distribution is known, the transform rescales the
    samples so that the mean of the output is the same as the mean of the
    input in expectation. If the mean of the distribution is not known, then
    the empirical mean of the sample is used instead. This is expected to help
    with consistency between training and inference."""
    func.__doc__ = func.__doc__.format(
        base_param_spec=base_param_spec,
        base_warning=base_warning,
        axial_param_spec=axial_param_spec,
        iid_scalar_param_spec=iid_scalar_param_spec,
        iid_tensor_param_spec=iid_tensor_param_spec,
        multiplicative_mean_correction=multiplicative_mean_correction
    )
    return func


def sample_multivariate(
    *,
    distr: Distribution,
    shape: Tuple[int],
    event_axes: Sequence[int],
    mean_correction: bool = False,
    key: jax.random.PRNGKey
):
    ndim = len(shape)
    event_axes = tuple(
        [standard_axis_number(axis, ndim) for axis in event_axes])
    event_shape = tuple([shape[axis] for axis in event_axes])
    sample_shape = tuple([shape[axis] for axis in range(ndim)
                          if axis not in event_axes])

    # This doesn't play well with JIT compilation.
    # if distr.event_shape != event_shape:
    #     raise ValueError(
    #         f"Distribution event shape {distr.event_shape} does not match "
    #         f"tensor shape {shape} along axes {event_axes}."
    #     )
    val = distr.sample(seed=key, sample_shape=sample_shape)

    if mean_correction:
        try:
            correction = 1 / (distr.mean() + jnp.finfo(jnp.float32).eps)
        except AttributeError:
            correction = 1 / (val.mean() + jnp.finfo(jnp.float64).eps)
        val = val * correction

    axis_order = argsort(axis_complement(ndim, event_axes) + event_axes)
    return jnp.transpose(val, axes=axis_order)


class StochasticSource(eqx.Module):
    """
    A wrapper for a pseudo-random number generator key.

    In general, this is not exposed to the user. It is embedded in other
    modules, such as ``StochasticTransform`` and its subclasses.

    If your training loop includes stochastic sources, you should use
    the :func:`refresh`` function to generate a new key for each epoch (or
    as needed). Each ``StochasticSource`` instance also has a ``code`` field
    that can be used to selectively refresh only a subset of the sources.
    In most cases, it is not necessary to change the default value of
    ``code``.
    """

    key: jax.random.PRNGKey
    code: Any = 0

    def refresh(self):
        """
        If you are calling this method, you are probably doing something
        wrong. You should probably use the :func:`refresh` function instead.
        """
        key = jax.random.split(self.key, 1)[0]
        return StochasticSource(key=key)


def _refresh_srcs(src: Any, code: Any = 0) -> Any:
    """
    Filter function for a PyTree that refreshes selected stochastic sources.
    """
    if _is_stochastic_source(src) and src.code == code:
        out = src.refresh()
    else:
        out = None
    return out


def _is_stochastic_source(src: Any) -> bool:
    return isinstance(src, StochasticSource)


def refresh(model: PyTree, code: Any = 0) -> PyTree:
    """
    Refresh all stochastic sources in a model.

    This ensures that each stochastic source has a new random number generator
    key and any stochastic transforms that depend on it will produce fresh
    samples.

    Parameters
    ----------
    model : PyTree
        The model to refresh.
    code : Any, optional
        The code to use when filtering stochastic sources. The default code
        is 0. In nearly all cases, it is not necessary to change the default
        value.

    Returns
    -------
    PyTree
        The model with all stochastic sources refreshed.
    """
    stochastic_srcs = eqx.filter(
        model,
        filter_spec=_is_stochastic_source,
        is_leaf=_is_stochastic_source
    )
    stochastic_srcs = jax.tree_util.tree_map(
        lambda x: _refresh_srcs(x, code=code),
        stochastic_srcs,
        is_leaf=_is_stochastic_source
    )
    return eqx.apply_updates(model, stochastic_srcs)


@document_stochastic_transforms
class StochasticTransform(eqx.Module):
    """
    Base class for stochastic transforms.

    Each stochastic transform must implement the following methods:

    - ``sample``: Sample from the noise distribution. Required keyword
                  parameters: ``key``, ``shape``.
    - ``inject``: Inject noise into a tensor. Required positional parameter:
                  ``input``. Required keyword parameters: ``key``.

    A stochastic transform on its own can be used as a layer for noise
    injection, regularisation or dropout (multiplicative noise). Its
    ``__call__`` (forward) method will sample from the noise distribution and
    inject it into the input depending on the value of the transform's
    ``inference`` attribute.

    Alternatively, it can be used to wrap a parameter of a model using the
    :class:``StochasticParameter`` API.

    Parameters
    ----------\
    {base_param_spec}
    """

    inference: bool = False
    source: StochasticSource

    def __init__(
        self,
        *,
        inference: bool = False,
        key: jax.random.PRNGKey,
        refresh_code: Any = 0,
    ):
        self.inference = inference
        self.source = StochasticSource(key=key, code=refresh_code)

    @abstractmethod
    def sample(
        self,
        *,
        shape: Tuple[int, ...],
        key: jax.random.PRNGKey
    ) -> Tensor:
        """
        Sample from the source.
        """
        raise NotImplementedError

    @abstractmethod
    def inject(
        self,
        input: Tensor,
        *,
        key: jax.random.PRNGKey
    ):
        """
        Inject stochasticity into the input.
        """
        raise NotImplementedError

    def __call__(self, input: Tensor) -> Tensor:
        if self.inference:
            return input
        return self.inject(input, key=self.source.key)


class StochasticParameter(eqx.Module):
    """
    A wrapper for a parameter that introduces stochasticity.

    .. note::
        Do not instantiate this class directly. Use the :method:`wrap` class
        method instead. This will return the model with the specified
        parameter wrapped in a :class:`StochasticTransform`.

    .. warning::
        When training models with stochastic parameters, you must apply the
        :func:`refresh` function to the parent model to ensure that each
        parameter has a fresh random number generator key. Otherwise, any
        stochastic transforms will produce the same stale samples across
        epochs.

    Parameters
    ----------
    model : PyTree
        The model to wrap.
    param_name : str
        The name of the parameter to wrap.
    transform : ``StochasticTransform``
        The transform to use for introducing stochasticity to the parameter.
        Inference mode and the kind of stochasticity are configured at the
        level of the transform.
    """

    original: Tensor
    transform: StochasticTransform

    def __init__(
        self,
        model: PyTree,
        *,
        where: Callable = where_weight,
        transform: StochasticTransform,
    ):
        self.original = where(model)
        self.transform = transform

    def __jax_array__(self):
        return self.transform(_to_jax_array(self.original))

    @classmethod
    def wrap(
        cls,
        model: PyTree,
        *pparams,
        param_name: str = "weight",
        transform: StochasticTransform,
        **params,
    ) -> PyTree:
        #TODO: We're inefficiently making a lot of repeated calls to
        #      ``retrieve_parameter`` here. We might be able to do this more
        #      efficiently, but this is low-priority as each call usually has
        #      very little overhead.
        parameters = retrieve_parameter(model, param_name)
        wrapped = ()
        for i, _ in enumerate(parameters):
            where = lambda model: retrieve_parameter(model, param_name)[i]
            wrapped += (cls(
                model=model,
                *pparams,
                where=where,
                transform=transform,
                **params),)
        return eqx.tree_at(
            lambda m: retrieve_parameter(m, param_name),
            model,
            replace=wrapped,
        )


class AdditiveNoiseMixin:
    """
    Mixin for configuring a ``StochasticTransform`` to add noise to the input.

    This implements the ``inject`` method for injecting noise additively.
    Any subclasses are still responsible for implementing the ``sample``
    method.
    """
    def inject(self, input: Tensor, *, key: jax.random.PRNGKey) -> Tensor:
        return input + self.sample(shape=input.shape, key=key)


class MultiplicativeNoiseMixin:
    """
    Mixin for configuring a ``StochasticTransform`` to multiply the input
    with noise (i.e., apply dropout).

    This implements the ``inject`` method for injecting noise
    multiplicatively. Any subclasses are still responsible for implementing
    the ``sample`` method.
    """
    def inject(self, input: Tensor, *, key: jax.random.PRNGKey) -> Tensor:
        return input * self.sample(shape=input.shape, key=key)


class ConvexCombinationNoiseMixin:
    """
    Mixin for configuring a ``StochasticTransform`` to combine the input with
    noise as a convex combination.

    This implements the ``inject`` method for injecting noise as a convex
    combination. Any subclasses are still responsible for implementing the
    ``sample`` method.
    """
    def inject(self, input: Tensor, *, key: jax.random.PRNGKey) -> Tensor:
        return (
            (1 - self.c) * input +
            self.c * self.sample(shape=input.shape, key=key)
        )


class AxialSelectiveTransform(StochasticTransform):
    """
    Noise source for which it is possible to specify the tensor axes along
    which there is randomness.

    This implements the ``sample`` method as a wrapper for sampling from the
    noise distribution. Any subclasses are still responsible for implementing
    the ``inject`` method. Additionally, subclasses are responsible for
    implementing the ``sample_impl`` method, which is wrapped by ``sample``.
    """

    sample_axes : Tuple[int, ...] = None

    def __init__(
        self,
        *,
        sample_axes: Optional[Tuple[int, ...]] = None,
        inference: bool = False,
        key: jax.random.PRNGKey,
        refresh_code: Any = 0
    ):
        self.sample_axes = sample_axes
        super().__init__(
            inference=inference,
            key=key,
            refresh_code=refresh_code
        )

    @abstractmethod
    def sample_impl(
        self,
        *,
        shape: Tuple[int],
        key: jax.random.PRNGKey
    ) -> Tensor:
        raise NotImplementedError

    def sample(self, *, shape: Tuple[int], key: jax.random.PRNGKey) -> Tensor:
        shape = self.select_dimensions(shape)
        return self.sample_impl(shape=shape, key=key)

    def canonicalise_axis_numbers(
        self,
        ndim: int,
        axes: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[int, ...]:
        if axes is None:
            if self.sample_axes is None:
                return None
            axes = self.sample_axes
        return [
            standard_axis_number(axis, ndim=ndim)
            for axis in axes
        ]

    def select_dimensions(self, shape: Tuple[int]) -> Tuple[int, ...]:
        """
        Change the dimension of the sample such that randomness occurs only
        along the specified axes.
        """
        if self.sample_axes is not None:
            ndim = len(shape)
            sample_axes = self.canonicalise_axis_numbers(
                ndim, self.sample_axes)
            shape = tuple([
                shape[axis] if axis in sample_axes else 1
                for axis in range(ndim)
            ])
        return shape


@document_stochastic_transforms
class ScalarIIDStochasticTransform(AxialSelectiveTransform):
    """
    Noise source where each sample is scalar-valued, independently drawn from
    an identical distribution.
    \
    {base_warning}

    Parameters
    ----------\
    {iid_scalar_param_spec}\
    {axial_param_spec}\
    {base_param_spec}
    """

    distribution : Distribution = eqx.static_field()

    def __init__(
        self,
        *,
        distribution: Distribution,
        sample_axes: Optional[Tuple[int, ...]] = None,
        inference: bool = False,
        key: jax.random.PRNGKey,
        refresh_code: Any = 0
    ):
        self.distribution = distribution
        super().__init__(
            sample_axes=sample_axes,
            inference=inference,
            key=key,
            refresh_code=refresh_code
        )

    def sample_impl(
        self,
        *,
        shape: Tuple[int],
        key: jax.random.PRNGKey
    ):
        return self.distribution.sample(seed=key, sample_shape=shape)


@document_stochastic_transforms
class TensorIIDStochasticTransform(AxialSelectiveTransform):
    """
    Noise source where each sample is multivariate, independently drawn from
    an identical distribution.
    \
    {base_warning}

    Parameters
    ----------\
    {iid_tensor_param_spec}\
    {axial_param_spec}\
    {base_param_spec}
    """

    distribution : Distribution = eqx.static_field()
    event_axes : Tuple[int, ...]

    def __init__(
        self,
        *,
        distribution: Distribution,
        event_axes: Optional[Tuple[int, ...]] = None,
        sample_axes: Optional[Tuple[int, ...]] = None,
        inference: bool = False,
        key: jax.random.PRNGKey,
        refresh_code: Any = 0
    ):
        self.distribution = distribution
        self.event_axes = event_axes
        super().__init__(
            sample_axes=sample_axes,
            inference=inference,
            key=key,
            refresh_code=refresh_code
        )

    def sample_impl(
        self,
        *,
        shape: Tuple[int],
        key: jax.random.PRNGKey
    ):
        event_axes = self.canonicalise_axis_numbers(
            ndim=len(shape), axes=self.event_axes)
        return sample_multivariate(
            distr=self.distribution,
            shape=shape,
            event_axes=event_axes,
            key=key
        )


@document_stochastic_transforms
class ScalarIIDAddStochasticTransform(
    AdditiveNoiseMixin,
    ScalarIIDStochasticTransform,
):
    """
    Additive noise source where each sample is scalar-valued, independently
    drawn from an identical distribution.
    \
    {base_warning}

    Parameters
    ----------\
    {iid_scalar_param_spec}\
    {axial_param_spec}\
    {base_param_spec}
    """


@document_stochastic_transforms
class TensorIIDAddStochasticTransform(
    AdditiveNoiseMixin,
    TensorIIDStochasticTransform,
):
    """
    Additive noise source where each sample is multivariate, independently
    drawn from an identical distribution.
    \
    {base_warning}

    Parameters
    ----------\
    {iid_tensor_param_spec}\
    {axial_param_spec}\
    {base_param_spec}
    """


@document_stochastic_transforms
class ScalarIIDMulStochasticTransform(
    MultiplicativeNoiseMixin,
    ScalarIIDStochasticTransform,
):
    """
    Multiplicative noise source (i.e., dropout) where each sample is scalar-
    valued, independently drawn from an identical distribution.
    \
    {multiplicative_mean_correction}
    \
    {base_warning}

    Parameters
    ----------\
    {iid_scalar_param_spec}\
    {axial_param_spec}\
    {base_param_spec}
    """
    def sample_impl(
        self,
        *,
        shape: Tuple[int],
        key: jax.random.PRNGKey
    ):
        sample = super().sample_impl(shape=shape, key=key)
        try:
            mean_correction = 1 / (
                self.distribution.mean() + jnp.finfo(jnp.float32).eps)
        except AttributeError:
            mean_correction = 1 / (
                sample.mean() + jnp.finfo(jnp.float32).eps)
        return mean_correction * sample


@document_stochastic_transforms
class TensorIIDMulStochasticTransform(
    MultiplicativeNoiseMixin,
    TensorIIDStochasticTransform,
):
    """
    Multiplicative noise source (i.e., dropout) where each sample is
    multivariate, independently drawn from an identical distribution.
    \
    {multiplicative_mean_correction}
    \
    {base_warning}

    Parameters
    ----------\
    {iid_scalar_param_spec}\
    {axial_param_spec}\
    {base_param_spec}
    """
    def sample_impl(
        self,
        *,
        shape: Tuple[int],
        key: jax.random.PRNGKey
    ):
        event_axes = self.canonicalise_axis_numbers(
            ndim=len(shape), axes=self.event_axes)
        return sample_multivariate(
            distr=self.distribution,
            shape=shape,
            event_axes=event_axes,
            mean_correction=True,
            key=key
        )


@document_stochastic_transforms
class EigenspaceReconditionTransform(
    TensorIIDAddStochasticTransform
):
    """
    Stochastic transform for reconditioning eigenspaces such there is neither
    degeneracy nor singularity.

    Differentiating through various numerical operations in linear algebra,
    such as the singular value decomposition (SVD), often suffers from
    numerical instability when the eigenvalues are close to degenerate
    because the vector-Jacobian product includes terms that depend on the
    reciprocal of the difference between eigenvalues. To mitigate this
    instability while yielding a decomposition that is still approximately
    correct, this transform stochastically introduces noise along the
    diagonal of the input matrix. Each input matrix A is transformed
    following:

    :math:`A := A + \\left(\\psi - \\frac{{\\xi}}{{2}}\\right) I + I\\mathbf{{x}}`

    :math:`x_i \\sim \\mathrm{{Uniform}}(0, \\xi) \\forall x_i`

    :math:`\\psi > \\xi`
    \
    {base_warning}

    Parameters
    ----------
    psi: float
        Reconditioning parameter to promote nonsingularity.
    xi: float
        Reconditioning parameter to promote nondegeneracy.
    matrix_dim: int
        Dimension of the square matrices to be reconditioned.
    event_axes : Tuple[int, ...] (default: (-2, -1))
        Specifies the axes of the input tensor that correspond to the event
        shape of the distribution (i.e., the slices along which matrices to be
        reconditioned lie). Note that event axes are automatically excluded
        from the sample axes.\
    {axial_param_spec}\
    {base_param_spec}
    """
    def __init__(
        self,
        *,
        psi: float,
        xi: float = None,
        matrix_dim: int,
        event_axes: Optional[Tuple[int, ...]] = (-2, -1),
        sample_axes: Optional[Tuple[int, ...]] = None,
        inference: bool = False,
        key: jax.random.PRNGKey,
        refresh_code: Any = 0
    ):
        if xi is None:
            xi = psi
        src_distribution = distrax.Uniform(low=(psi - xi), high=psi)
        distribution = Diagonal(
            src_distribution=src_distribution,
            multiplicity=matrix_dim,
        )
        super().__init__(
            distribution=distribution,
            event_axes=event_axes,
            sample_axes=sample_axes,
            inference=inference,
            key=key,
            refresh_code=refresh_code,
        )


class OuterProduct(distrax.Distribution):
    r"""
    Outer-product transformed distribution.

    This distribution ingests samples from a source distribution and
    computes their outer product to produce a square, symmetric, and
    positive semidefinite sample.

    The dimensions of the sample from the source distribution are equal to
    :math:`(E \cdot M) \times R`, where E is the event shape for the source
    distribution (1 for a univariate distribution), M is a user-specified
    multiplicity factor, and R is the specified matrix rank. The rank-R outer
    product of this sample with itself is then computed to produce the output.

    .. note::
        To obtain an outer-product distribution whose entries have
        approximately some approximate expected standard deviation, use a
        normal distribution for  the source after calling the static method
        ``rescale_std_for_normal`` to compute the standard deviation for
        the source distribution.

        For the transformed entries to have a standard deviation near
        :math:`\sigma`, each entry in the source sample is distributed as

        :math:`\mathcal{N}\left(0, \frac{\sigma}{\sqrt{r + \frac{r^2}{d}}}\right)`

        Note that the variance and mean for this transformation in fact
        belong to separate distributions for the on-diagonal and off-siagonal
        entries. On-diagonal entries are formed as quadratic sums and will
        have greater variance and mean.

    Parameters
    ----------
    src_distribution: Distribution
        The source distribution to sample from.
    rank: int (default 1)
        Matrix rank. If this is equal to the product of the multiplicity and
        the source distribution's event shape, then the result will be a
        positive definite matrix. Otherwise, it will be positive semidefinite.
    multiplicity: int (default 1)
        The multiplicity factor for source distribution samples.
    """
    def __init__(
        self,
        src_distribution: Distribution,
        rank: int = 1,
        multiplicity: int = 1,
    ):
        self.src_distribution = src_distribution
        self.rank = rank
        self.multiplicity = multiplicity
        self.matrix_dim = self.multiplicity * jnp.prod(
            jnp.asarray(self.src_distribution.event_shape, dtype=int))
        super().__init__()

    def _sample_n(self, key: jax.random.PRNGKey, n: int) -> Tensor:
        samples = self.src_distribution.sample(
            seed=key, sample_shape=(n, self.rank, self.multiplicity))
        samples = samples.reshape((n, self.rank, -1))
        samples = samples.swapaxes(-1, -2) @ samples
        return samples

    def log_prob(self, value: Tensor) -> Tensor:
        # There's not a systematic way to work this out, and it's nontrivial
        # to figure out even for a simple distribution, e.g.
        # https://math.stackexchange.com/questions/101062/ ...
        #     is-the-product-of-two-gaussian-random-variables-also-a-gaussian
        # -- and that's only for off-diagonals.
        #
        # So, we just return NaN.
        return float('nan') * value

    @property
    def event_shape(self):
        return (self.matrix_dim, self.matrix_dim)

    def mean(self):
        mu = jnp.atleast_1d(self.src_distribution.mean())
        src_mean = jnp.concatenate(
            (mu,) * self.multiplicity
        )[..., None]
        return self.rank * src_mean @ src_mean.T

    @staticmethod
    def rescale_std_for_normal(
        std: Tensor,
        rank: int,
        matrix_dim: int
    ) -> Tensor:
        """
        Find the standard deviation of a normal distribution such that the
        outer product of its samples has approximately a specified standard
        deviation.

        This is a bit of a hack and isn't correct, but it's the best we can do
        given we have more pressing problems to solve. In reality, the
        diagonal entries tend to have a larger standard deviation than the
        off-diagonal entries.

        The output of this static method is intended to be used to initialise
        the standard deviation of a normal distribution that can then be used
        as the ``src_distribution`` argument to an ``OuterProduct``
        distribution.

        Parameters
        ----------
        std : Tensor
            The desired standard deviation of the outer product of the samples.
        rank : int
            The rank of the samples.
        matrix_dim : int
            The dimension of the square matrices output by the outer product.
        """
        return math.sqrt(std / math.sqrt(rank + (rank ** 2) / matrix_dim))


class Diagonal(distrax.Distribution):
    """
    Square diagonal transformed distribution.

    This distribution ingests samples from a source distribution and embeds
    them along the diagonal of its outputs. ``multiplicity`` samples are drawn
    from the source distribution, each with shape ``event_shape`` (1 for
    univariate/scalar distributions). The dimension of each output event is
    thereby the product of ``multiplicity`` and ``event_shape``.

    Parameters
    ----------
    src_distribution: Distribution
        The source distribution to sample from.
    multiplicity: int (default 1)
        The multiplicity factor for source distribution samples.
    """
    def __init__(
        self,
        src_distribution: Distribution,
        multiplicity: int = 1,
    ):
        self.src_distribution = src_distribution
        self.multiplicity = multiplicity
        self.matrix_dim = self.multiplicity * jnp.prod(
            jnp.asarray(self.src_distribution.event_shape, dtype=int))
        super().__init__()

    def _sample_n(self, key: jax.random.PRNGKey, n: int) -> Tensor:
        samples = self.src_distribution.sample(
            seed=key, sample_shape=(n, self.multiplicity))
        samples = samples.reshape((n, -1))
        samples = jax.vmap(jnp.diagflat, in_axes=(0,))(samples)
        return samples

    def log_prob(self, value: Tensor) -> Tensor:
        # We set the log probability to 0 for any value that is not on the
        # diagonal, because it is zero with expectation 1.
        mask = jnp.eye(self.matrix_dim, dtype=jnp.bool_)[None, ...]
        diags = jnp.diagonal(value, axis1=-2, axis2=-1)[..., None]
        log_prob = self.src_distribution.log_prob(diags)
        return jnp.where(mask, log_prob, 0.)

    @property
    def event_shape(self):
        return (self.matrix_dim, self.matrix_dim)

    def mean(self):
        mu = jnp.atleast_1d(self.src_distribution.mean())
        return jnp.concatenate(
            (mu,) * self.multiplicity
        ) * jnp.eye(self.matrix_dim)


class Symmetric(distrax.Distribution):
    def __init__(
        self,
        src_distribution: Distribution,
        multiplicity: int = 1,
    ):
        self.src_distribution = src_distribution
        self.multiplicity = multiplicity
        self.matrix_dim = self.multiplicity * jnp.prod(
            jnp.asarray(self.src_distribution.event_shape, dtype=int))
        self._determine_shape()
        super().__init__()

    def _determine_shape(self):
        src_event_shape = self.src_distribution.event_shape
        if len(src_event_shape) > 2:
            raise ValueError('Invalid source distribution was provided: '
                             'event_shape must be a scalar or a 1- or 2-D '
                             'tensor.')
        elif len(src_event_shape) == 0:
            self.matrix_dim = self.multiplicity
            self.row_multiplicity = self.col_multiplicity = self.multiplicity
        elif len(src_event_shape) == 1:
            self.matrix_dim = self.multiplicity * src_event_shape[0]
            self.row_multiplicity = self.multiplicity
            self.col_multiplicity = self.multiplicity * src_event_shape[0]
        else:
            self.matrix_dim = self.multiplicity * src_event_shape[-1]
            self.row_multiplicity = self.multiplicity
            self.col_multiplicity = self.matrix_dim // src_event_shape[0]

    def _sample_n(self, key: jax.random.PRNGKey, n: int) -> Tensor:
        src_event_shape = self.src_distribution.event_shape
        samples = self.src_distribution.sample(
            seed=key, sample_shape=(
                n, self.row_multiplicity, self.col_multiplicity)
        )
        if len(src_event_shape) == 2:
            samples = samples.swapaxes(-3, -2)
        samples = samples.reshape(n, self.matrix_dim, -1)
        return samples + samples.swapaxes(-2, -1)

    def log_prob(self, value: Tensor) -> Tensor:
        # In the general continuous case we need to do a convolution, but we
        # might not have access to the PDFs, and there's no guarantee that the
        # source distribution is continuous anyway, so we're leaving this as
        # NaN
        return float('nan') * value

    @property
    def event_shape(self):
        return (self.matrix_dim, self.matrix_dim)

    def mean(self):
        return 2 * self.src_distribution.mean()


class MatrixExponential(distrax.Distribution):
    """
    Matrix exponential transformed distribution.

    This distribution ingests square matrix samples from a source distribution
    and then projects them into the positive semidefinite cone by way of the
    matrix exponential.

    .. warning::
        This transformation can substantially scale up the values sampled from
        the input distribution. Overflow is possible if rescaling the variance
        automatically. If you wish to do this, it is recommended to keep the
        input distribution small.

    Parameters
    ----------
    src_distribution: Distribution
        The source distribution to sample from.
    rescale_var: bool (default: True)
        Indicates that the entry-wise variance of the transformed samples
        should be rescaled empirically to match the entry-wise variance of the
        source samples.
    """
    def __init__(
        self,
        src_distribution: Distribution,
        rescale_var: bool = True,
    ):
        self.src_distribution = src_distribution
        self.rescale_var = rescale_var
        super().__init__()

    def _sample_n(self, key: jax.random.PRNGKey, n: int) -> Tensor:
        samples = self.src_distribution.sample(
            seed=key, sample_shape=(n,))
        if self.rescale_var:
            var_orig = samples.var(keepdims=True)
        samples = jax.vmap(jax.scipy.linalg.expm, in_axes=(0,))(samples)
        if self.rescale_var:
            var_transformed = samples.var(keepdims=True)
            samples = samples / jnp.sqrt(var_transformed / var_orig)
        return samples

    def log_prob(self, value: Tensor) -> Tensor:
        samples = _symlog(value)
        return self.src_distribution.log_prob(samples)

    def _sample_n_and_log_prob(
        self, key: jax.random.PRNGKey, n: int
    ) -> Tensor:
        samples = self.src_distribution.sample(
            seed=key, sample_shape=(n,))
        log_prob = self.src_distribution.log_prob(samples)
        samples = jax.vmap(jax.scipy.linalg.expm, in_axes=(0,))(samples)
        return samples, log_prob

    @property
    def event_shape(self):
        return (self.matrix_dim, self.matrix_dim)


# test import compatibility
class _IIDSource:
    def __init__():
        raise NotImplementedError()

class _IIDNoiseSource:
    def __init__():
        raise NotImplementedError()

class _IIDSquareNoiseSource:
    def __init__():
        raise NotImplementedError()

class _IIDDropoutSource:
    def __init__():
        raise NotImplementedError()

class _IIDSquareDropoutSource:
    def __init__():
        raise NotImplementedError()

class _AxialSampler:
    def __init__():
        raise NotImplementedError()

class UnstructuredNoiseSource:
    def __init__():
        raise NotImplementedError()

class DiagonalNoiseSource:
    def __init__():
        raise NotImplementedError()

class LowRankNoiseSource:
    def __init__():
        raise NotImplementedError()

class SPSDNoiseSource:
    def __init__():
        raise NotImplementedError()

class UnstructuredDropoutSource:
    def __init__():
        raise NotImplementedError()

class DiagonalDropoutSource:
    def __init__():
        raise NotImplementedError()

class BandDropoutSource:
    def __init__():
        raise NotImplementedError()

class SPSDDropoutSource:
    def __init__():
        raise NotImplementedError()

class IdentitySource:
    def __init__():
        raise NotImplementedError()

class IdentityNoiseSource:
    def __init__():
        raise NotImplementedError()

class IdentityDropoutSource:
    def __init__():
        raise NotImplementedError()
