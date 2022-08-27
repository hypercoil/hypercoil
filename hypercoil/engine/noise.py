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
from typing import Any, Optional, Tuple
from ..functional.utils import (
    Distribution, PyTree, Tensor,
    sample_multivariate, standard_axis_number
)
from ..functional.symmap import symlog
from ..init.mapparam import _to_jax_array


#TODO: eqx.filter isn't compatible with distrax.Distribution objects.
#      To get around this, we currently set distributions as static fields.
#      This is a hack, but it should work for most of our needs as long as
#      the distribution is the same instance across epoch updates.
#      In the long run, we will want to figure out the cause for this
#      incompatibility.
#
#      Opened an issue on distrax:
#      https://github.com/deepmind/distrax/issues/193


class StochasticSource(eqx.Module):
    """
    A wrapper for a pseudo-random number generator key.
    """

    key: jax.random.PRNGKey
    code: Any = 0

    def refresh(self):
        key = jax.random.split(self.key, 1)[0]
        return StochasticSource(key=key)


def _refresh_srcs(src: Any, code: Any = 0) -> Any:
    if _is_stochastic_source(src) and src.code == code:
        out = src.refresh()
    else:
        out = None
    return out


def _is_stochastic_source(src: Any) -> bool:
    return isinstance(src, StochasticSource)


def refresh(model: PyTree, code: Any = 0) -> PyTree:
    # We have to set the distributions as leaves. I'm not sure why this is.
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


class StochasticTransform(eqx.Module):

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


class AdditiveNoiseMixin:
    def inject(self, input: Tensor, *, key: jax.random.PRNGKey) -> Tensor:
        return input + self.sample(shape=input.shape, key=key)


class MultiplicativeNoiseMixin:
    def inject(self, input: Tensor, *, key: jax.random.PRNGKey) -> Tensor:
        return input * self.sample(shape=input.shape, key=key)


class ConvexCombinationNoiseMixin:
    def inject(self, input: Tensor, *, key: jax.random.PRNGKey) -> Tensor:
        return (
            (1 - self.c) * input +
            self.c * self.sample(shape=input.shape, key=key)
        )


class AxialSelectiveTransform(StochasticTransform):
    """
    Noise source for which it is possible to specify the tensor axes along
    which there is randomness.
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


class ScalarIIDStochasticTransform(AxialSelectiveTransform):
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


class TensorIIDStochasticTransform(AxialSelectiveTransform):
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


class ScalarIIDAddStochasticTransform(
    AdditiveNoiseMixin,
    ScalarIIDStochasticTransform,
):
    pass


class TensorIIDAddStochasticTransform(
    AdditiveNoiseMixin,
    TensorIIDStochasticTransform,
):
    pass


class ScalarIIDMulStochasticTransform(
    MultiplicativeNoiseMixin,
    ScalarIIDStochasticTransform,
):
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


class TensorIIDMulStochasticTransform(
    MultiplicativeNoiseMixin,
    TensorIIDStochasticTransform,
):
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


class OuterProduct(distrax.Distribution):
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

    def _sample_n(self, key, n):
        samples = self.src_distribution.sample(
            seed=key, sample_shape=(n, self.rank, self.multiplicity))
        samples = samples.reshape((n, self.rank, -1))
        samples = samples.swapaxes(-1, -2) @ samples
        return samples

    def log_prob(self, value):
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

    def _sample_n(self, key, n):
        samples = self.src_distribution.sample(
            seed=key, sample_shape=(n, self.multiplicity))
        samples = samples.reshape((n, -1))
        samples = jax.vmap(jnp.diagflat, in_axes=(0,))(samples)
        return samples

    def log_prob(self, value):
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


class MatrixExponential(distrax.Distribution):
    def __init__(
        self,
        src_distribution: Distribution,
        rescale_var: bool = True,
    ):
        self.src_distribution = src_distribution
        self.rescale_var = rescale_var
        super().__init__()

    def _sample_n(self, key, n):
        samples = self.src_distribution.sample(
            seed=key, sample_shape=(n,))
        if self.rescale_var:
            var_orig = samples.var(keepdims=True)
        samples = jax.vmap(jax.scipy.linalg.expm, in_axes=(0,))(samples)
        if self.rescale_var:
            var_transformed = samples.var(keepdims=True)
            samples = samples / jnp.sqrt(var_transformed / var_orig)
        return samples

    def log_prob(self, value):
        samples = symlog(value)
        return self.src_distribution.log_prob(samples)

    def _sample_n_and_log_prob(self, key, n):
        samples = self.src_distribution.sample(
            seed=key, sample_shape=(n,))
        log_prob = self.src_distribution.log_prob(samples)
        samples = jax.vmap(jax.scipy.linalg.expm, in_axes=(0,))(samples)
        return samples, log_prob

    @property
    def event_shape(self):
        return (self.matrix_dim, self.matrix_dim)


class _IIDSource():
    """
    Superclass for i.i.d. noise and dropout sources. Implements methods that
    toggle between test and train mode.

    There's nothing about this implementation level that requires the i.i.d.
    assumption.
    """
    def __init__():
        raise NotImplementedError()

class _IIDNoiseSource():
    """
    Superclass for i.i.d. noise sources. Implements a method for injecting
    sampled noise additively into an existing tensor.

    Subclasses are responsible for implementing the correct `sample` method
    that accepts a dimension argument.

    See also
    --------
    _IIDSquareNoiseSource : Noise source with a square matrix assumption.
    _IIDDropoutSource : For multiplicative sample injection.
    """
    def __init__():
        raise NotImplementedError()

class _IIDSquareNoiseSource():
    """
    Superclass for i.i.d. noise sources. Implements a method for injecting
    sampled noise additively into an existing tensor.

    Subclasses are responsible for implementing the correct `sample` method
    that accepts a dimension argument. For use when there is a square matrix
    input assumption in the `inject` method.

    See also
    --------
    _IIDNoiseSource : Does not assume samples are square matrices.
    _IIDSquareDropoutSource : For multiplicative sample injection.
    """
    def __init__():
        raise NotImplementedError()

class _IIDDropoutSource():
    """
    Superclass for i.i.d. noise sources. Implements a method for injecting
    sampled noise multiplicatively into an existing tensor.

    Subclasses are responsible for implementing the correct `sample` method
    that accepts a dimension argument.

    See also
    --------
    _IIDSquareDropoutSource : Dropout source with a square matrix assumption.
    _IIDNoiseSource : For additive sample injection.
    """
    def __init__():
        raise NotImplementedError()

class _IIDSquareDropoutSource():
    """
    Superclass for i.i.d. noise sources. Implements a method for injecting
    sampled noise multiplicatively into an existing tensor.

    Subclasses are responsible for implementing the correct `sample` method
    that accepts a dimension argument. Currently there is a square matrix
    input assumption in the `inject` method that future work might revise.

    See also
    --------
    _IIDDropoutSource : Does not assume samples are square matrices.
    _IIDSquareNoiseSource : For additive sample injection.
    """
    def __init__():
        raise NotImplementedError()

class _AxialSampler():
    """
    Noise source for which it is possible to specify the tensor axes along
    which there is randomness.
    """
    def __init__():
        raise NotImplementedError()

class UnstructuredNoiseSource():
    """
    Additive noise source with no special structure, in which each element is
    sampled i.i.d.

    Parameters
    ----------
    distr : torch.distributions object
        Distribution from which each element is sampled independently. If not
        specified, this defaults to the standard normal distribution.
    training : bool
        Indicates whether the source should operate under the assumption of
        training or inference; at test time, a noise-free sample is returned.
    sample_axes : list or None (default None)
        Axes along which sampling is performed. Along all other axes, the same
        samples are broadcast. If this is None, then sampling occurs along all
        axes.
    """
    def __init__():
        raise NotImplementedError()

class DiagonalNoiseSource():
    """
    Diagonal noise source.

    Parameters
    ----------
    distr : torch.distributions object
        Distribution from which each element is sampled independently. If not
        specified, this defaults to the standard normal distribution.
    offset : int (default 0)
        Diagonal along which the noise is to be embedded. The default value of
        0 corresponds to the main diagonal of the matrix; positive values
        indicate upper diagonals and negative values indicate lower diagonals.
    training : bool
        Indicates whether the source should operate under the assumption of
        training or inference; at test time, a noise-free sample is returned.
    """
    def __init__():
        raise NotImplementedError()

class LowRankNoiseSource():
    """
    Low-rank symmetric positive semidefinite noise source. Note that diagonal
    entries are effectively sampled from a very different distribution than
    off-diagonal entries. Be careful using this source; examine outputs for the
    dimension that you will be using before using it, as it exhibits some
    potentially very undesirable properties, particularly when the rank becomes
    larger.

    Parameters
    ----------
    var : torch.distributions object
        Average variance across entries of the output matrix.
    rank : int or None (default None)
        Maximum rank of each sampled matrix; inner dimension of the positive
        semidefinite product. If this is less than the sampled dimension, the
        sampled matrices will be singular; if it is greater or equal, they will
        likely be positive definite. Regardless of the value of this parameter,
        the output rank cannot be greater than the matrix dimension.
    training : bool
        Indicates whether the source should operate under the assumption of
        training or inference; at test time, a noise-free sample is returned.
    """
    def __init__():
        raise NotImplementedError()

class SPSDNoiseSource():
    """
    Symmetric positive semidefinite noise source.

    Uses the matrix exponential to project a symmetric noise matrix into the
    positive semidefinite cone. The symmetric matrix is diagonalised, its
    eigenvalues are exponentiated thereby ensuring each is positive, and it
    is recomposed. Note that due to numerical errors some extremely small
    negative eigenvalues can occur in the sampled matrix.

    Parameters
    ----------
    distr : torch.distributions object
        Distribution from which each element is sampled independently. If not
        specified, this defaults to the standard normal distribution.
    training : bool
        Indicates whether the source should operate under the assumption of
        training or inference; at test time, a noise-free sample is returned.
    """
    def __init__():
        raise NotImplementedError()

class UnstructuredDropoutSource():
    """
    Multiplicative noise source with no special structure, in which each
    element is sampled i.i.d.

    Parameters
    ----------
    distr : torch.distributions object
        Distribution from which each element is sampled independently. If not
        specified, this defaults to the standard normal distribution.
    training : bool
        Indicates whether the source should operate under the assumption of
        training or inference; at test time, a noise-free sample is returned.
    sample_axes : list or None (default None)
        Axes along which sampling is performed. Along all other axes, the same
        samples are broadcast. If this is None, then sampling occurs along all
        axes.
    """
    def __init__():
        raise NotImplementedError()

class DiagonalDropoutSource():
    """
    Diagonal dropout source.

    Parameters
    ----------
    distr : torch.distributions object
        Distribution from which each element is sampled independently. If not
        specified, this defaults to an equiprobable Bernoulli distribution.
    offset : int (default 0)
        Diagonal along which the mask is to be embedded. The default value of
        0 corresponds to the main diagonal of the matrix; positive values
        indicate upper diagonals and negative values indicate lower diagonals.
    training : bool
        Indicates whether the source should operate under the assumption of
        training or inference; at test time, a noise-free sample is returned.
    """
    def __init__():
        raise NotImplementedError()

class BandDropoutSource():
    def __init__():
        raise NotImplementedError()

class SPSDDropoutSource():
    """
    Symmetric positive semidefinite dropout source. Note that diagonal entries
    are effectively sampled from a very different distribution than
    off-diagonal entries. Be careful using this source; examine outputs for the
    dimension that you will be using before using it, as it exhibits some
    potentially very undesirable properties. This will be revisited in the
    future to determine if a better source can be provided.

    Parameters
    ----------
    distr : torch.distributions object
        Distribution from which each element is sampled independently. If not
        specified, this defaults to an equiprobable Bernoulli distribution. For
        a Bernoulli distribution, then its parameter corresponds to the
        approximate probability of dropout across columns in each symmetric
        positive semidefinite masking matrix sampled from the source.
    rank : int or None (default None)
        Maximum rank of each masking matrix; inner dimension of the positive
        semidefinite product. Only a rank of 1 will intuitively correspond to
        standard dropout; any other rank might not yield a strictly on/off
        masking matrix.
    training : bool
        Indicates whether the source should operate under the assumption of
        training or inference; at test time, a noise-free sample is returned.
    """
    def __init__():
        raise NotImplementedError()

class IdentitySource(_IIDSource):
    def __init__():
        raise NotImplementedError()

class IdentityNoiseSource(IdentitySource):
    def __init__():
        raise NotImplementedError()

class IdentityDropoutSource(IdentitySource):
    def __init__():
        raise NotImplementedError()
