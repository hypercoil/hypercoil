# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Noise
~~~~~
Additive and multiplicative noise sources.
"""
import math
import torch
from abc import ABC, abstractmethod
from ..functional.matrix import toeplitz

import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any, Optional, Tuple
from ..functional.utils import (
    Distribution, PyTree, Tensor,
    sample_multivariate, standard_axis_number
)
from ..init.mapparam import _to_jax_array


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
    if isinstance(src, StochasticSource) and src.code == code:
        out = src.refresh()
    else:
        out = None
    return out


def refresh(model: PyTree, code: Any = 0) -> PyTree:
    stochastic_srcs = eqx.filter(
        model,
        filter_spec=lambda x: isinstance(x, StochasticSource),
        is_leaf=lambda x: isinstance(x, StochasticSource)
    )
    stochastic_srcs = jax.tree_util.tree_map(
        lambda x: _refresh_srcs(x, code=code),
        stochastic_srcs,
        is_leaf=lambda x: isinstance(x, StochasticSource)
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
    distribution : Distribution

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
    distribution : Distribution
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

    def conform_scale_factor_to_shape(
        self,
        rescale: Tensor,
        shape: Tuple[int],
    ):
        """
        Does nothing. Sufficient for scalar events. Override in subclasses to
        conform the scale factor to the shape of the tensor.
        """
        return rescale

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
        try:
            mean_correction = 1 / (
                self.distribution.mean() + jnp.finfo(jnp.float32).eps)
        except AttributeError:
            mean_correction = 1
        return mean_correction * super().sample_impl(shape=shape, key=key)


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






class _IIDSource(torch.nn.Module, ABC):
    """
    Superclass for i.i.d. noise and dropout sources. Implements methods that
    toggle between test and train mode.

    There's nothing about this implementation level that requires the i.i.d.
    assumption.
    """
    def __init__(self, distr, training):
        raise NotImplementedError()


class _IIDNoiseSource(_IIDSource):
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
    def __init__(self, distr=None, training=True):
        raise NotImplementedError()


class _IIDSquareNoiseSource(_IIDNoiseSource):
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
    def inject(self, tensor):
        try:
            sz = tensor.size()
            assert sz[-1] == sz[-2]
        except AssertionError:
            raise AssertionError('Cannot inject square noise into nonsquare '
                                 'tensors. The tensors must be square along '
                                 'the last two dimensions.')
        if self.training:
            return tensor + self.sample(
                tensor.size()[:-1]).to(dtype=tensor.dtype, device=tensor.device)
        else:
            return tensor


class _IIDDropoutSource(_IIDSource):
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
    def __init__(self, distr=None, training=True):
        raise NotImplementedError()


class _IIDSquareDropoutSource(_IIDDropoutSource):
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
    def inject(self, tensor):
        try:
            sz = tensor.size()
            assert sz[-1] == sz[-2]
        except AssertionError:
            raise AssertionError('Cannot inject square noise into nonsquare '
                                 'tensors. The tensors must be square along '
                                 'the last two dimensions.')
        if self.training:
            return tensor * self.sample(
                tensor.size()[:-1]).to(
                    dtype=tensor.dtype, device=tensor.device)
        else:
            return tensor


class _AxialSampler(_IIDSource):
    """
    Noise source for which it is possible to specify the tensor axes along
    which there is randomness.
    """
    def __init__(self, distr=None, training=True, sample_axes=None):
        raise NotImplementedError()


class UnstructuredNoiseSource(_AxialSampler, _IIDNoiseSource):
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


class DiagonalNoiseSource(_IIDSquareNoiseSource):
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
    def __init__(self, distr=None, offset=0, training=True):
        super(DiagonalNoiseSource, self).__init__(distr, training)
        self.offset = offset

    def sample(self, dim):
        """
        Sample a random matrix that is zero everywhere except for a single
        diagonal band, along which the entries are sampled i.i.d. from the
        specified distribution.

        Parameters
        ----------
        dim : iterable
            Dimension of the matrices sampled from the source.

        Returns
        -------
        output : Tensor
            Block of diagonal matrices sampled from the noise source.
        """
        dim = [*dim[:-1], dim[-1] - abs(self.offset)]
        if self.training:
            noise = self.distr.sample(dim).squeeze(-1)
            return torch.diag_embed(noise, self.offset)
        else:
            return 0


class LowRankNoiseSource(_IIDNoiseSource):
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
    def __init__(self, var=1, rank=None, training=True):
        super(LowRankNoiseSource, self).__init__(
            distr=None, training=training)
        self.rank = rank
        self.var = var

    def sample(self, dim):
        r"""
        Sample a random matrix :math:`K \in \mathbb{R}^{d \times r}` and
        computes the rank-r positive semidefinite product :math:`KK^\intercal`.
        For the outcome entries to have standard deviation near :math:`\sigma`,
        each entry in K (sampled i.i.d.) is distributed as

        :math:`\mathcal{N}\left(0, \frac{\sigma}{\sqrt{r + \frac{r^2}{d}}}\right)`

        The mean of this noise source is not exactly zero, but it trends
        toward zero as the dimension d increases.

        Parameters
        ----------
        dim : iterable
            Dimension of the matrices sampled from the source.

        Returns
        -------
        output : Tensor
            Block of symmetric, positive semidefinite matrices sampled from
            the low-rank noise source.

        See also
        --------
        SPSDNoiseSource
            Another way to sample noise from the cone of symmetric, positive
            semidefinite matrices.
        """
        #TODO: revisit this later and work out what is going on
        # mathematically. Here's a start:
        # https://math.stackexchange.com/questions/101062/ ...
        # is-the-product-of-two-gaussian-random-variables-also-a-gaussian
        if self.training:
            rank = self.rank or dim[-1]
            noise = torch.empty((*dim, rank))
            var = self.var / math.sqrt(rank + (rank ** 2) / dim[-1])
            noise.normal_(std=math.sqrt(var))
            return noise @ noise.transpose(-1, -2)
        else:
            return 0


class SPSDNoiseSource(_IIDSquareNoiseSource):
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
    def sample(self, dim):
        """
        Sample a random symmetric positive (semi)definite matrix from the noise
        source.

        Parameters
        ----------
        dim : iterable
            Dimension of the matrices sampled from the source.

        Returns
        -------
        output : Tensor
            Block of symmetric, positive semidefinite matrices sampled from the
            noise source.
        """
        if self.training:
            noise = self.distr.sample((*dim, dim[-1])).squeeze(-1)
            sym = noise + noise.transpose(-1, -2)
            spd = torch.matrix_exp(sym)
            return spd / (spd.std() / self.distr.scale)
        else:
            return 0


class UnstructuredDropoutSource(_AxialSampler, _IIDDropoutSource):
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


class DiagonalDropoutSource(_IIDSquareDropoutSource):
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
    def __init__(self, distr=None, offset=0, training=True):
        super(DiagonalDropoutSource, self).__init__(distr, training)
        self.offset = offset

    def sample(self, dim):
        """
        Sample a random masking matrix from the dropout source.

        Parameters
        ----------
        dim : iterable
            Dimension of the matrices sampled from the source.

        Returns
        -------
        output : Tensor
            Block of masking matrices sampled from the noise source.
        """
        dim = [*dim[:-1], dim[-1] - abs(self.offset)]
        if self.training:
            mask = self.distr.sample(dim).squeeze(-1) / self.distr.mean
            return torch.diag_embed(mask, self.offset)
        else:
            return 1


class BandDropoutSource(_IIDSquareDropoutSource):
    """
    Dropout source for matrices with banded structure.

    This source applies dropout to a block of banded matrices (all nonzero
    entries within some finite offset of the main diagonal.) It creates a
    dropout mask by multiplying together the band mask with a dropout mask
    in which a random subset of rows and columns are zeroed.

    Parameters
    ----------
    distr : torch.distributions object
        Distribution from which each element is sampled independently. If not
        specified, this defaults to an equiprobable Bernoulli distribution. For
        a Bernoulli distribution, then its parameter corresponds to the
        approximate probability of dropout across columns in each symmetric
        positive semidefinite masking matrix sampled from the source.
    bandwidth : int or None (default None)
        Maximum bandwidth of each masking matrix; maximum offset from the main
        diagonal where nonzero entries are permitted. 0 indicates that the
        matrix is diagonal (in which case `DiagonalDropoutSource` is faster and
        more efficient).
    training : bool
        Indicates whether the source should operate under the assumption of
        training or inference; at test time, a noise-free sample is returned.
    norm : 'blanket' or 'diag' (default `diag`)
        Entries along the main diagonal are sampled from a different
        distribution compared with off-diagonal entries. In particular, the
        probability of each entry along the diagonal surviving dropout is
        1 - p, while the probability of each off diagonal entry surviving
        dropout is :math:`(1 - p)^2`. This parameter indicates whether the same
        correction term is applied to both on-diagonal and off-diagonal entries
        (`blanket`) or whether a separate correction term is applied to
        diagonal and off-diagonal entries after dropout.
    generator, bandmask, bandnorm
        Attributes related to a Boolean mask tensor indicating whether each
        entry is in the permitted band.
    normfact : float or Tensor
        Dropout correction term.
    """
    def __init__(self, distr=None, bandwidth=0, training=True, norm='diag'):
        super(BandDropoutSource, self).__init__(distr, training)
        self.generator = torch.Tensor([1] * (1 + bandwidth))
        self.bandwidth = bandwidth
        self.n = float('nan')
        self.norm = norm

    def _create_bandmask(self, n):
        self.n = n
        self.bandmask = toeplitz(self.generator, dim=[self.n, self.n])
        self.bandnorm = self.bandmask.sum()
        if self.norm == 'blanket':
            self.normfact = self.bandnorm / (self.n * (1 - self.distr.mean) +
                (self.bandnorm - self.n) * (1 - self.distr.mean) ** 2)
        elif self.norm == 'diag':
            self.normfact = (torch.ones_like(self.bandmask) /
                             (1 - self.distr.mean) ** 2)
            self.normfact[torch.eye(self.n).bool()] = 1 / (1 - self.distr.mean)

    def sample(self, dim):
        """
        Sample a random masking matrix from the dropout source.

        Parameters
        ----------
        dim : iterable
            Dimension of the matrices sampled from the source. Note that, every
            time the source is queried to sample matrices of a different
            dimension, there will be a delay as the band mask is rebuilt. If
            you are sampling repeatedly from sources with multiple sizes, it is
            thus more time-efficient to use multiple BandDropoutSources.

        Returns
        -------
        output : Tensor
            Block of masking matrices sampled from the noise source.
        """
        if self.training:
            n = dim[-1]
            if n != self.n:
                self._create_bandmask(n)
            mask = self.distr.sample((*dim, 1))
            unnorm = mask @ mask.transpose(-1, -2) * self.bandmask
            return unnorm * self.normfact
        else:
            return 1


class SPSDDropoutSource(_IIDSquareDropoutSource):
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
    def __init__(self, distr=None, rank=1, training=True):
        super(SPSDDropoutSource, self).__init__(distr, training)
        self.rank = rank

    def sample(self, dim):
        """
        Sample a random masking matrix from the dropout source.

        Parameters
        ----------
        dim : iterable
            Dimension of the matrices sampled from the source.

        Returns
        -------
        output : Tensor
            Block of masking matrices sampled from the noise source.
        """
        if self.training:
            rank = self.rank or dim[-1]
            mask = self.distr.sample(dim).squeeze(-1) / self.distr.mean
            return mask @ mask.transpose(-1, -2) / rank
        else:
            return 1


class IdentitySource(_IIDSource):
    def __init__(self, training=True):
        raise NotImplementedError()

class IdentityNoiseSource(IdentitySource):
    def __init__(self, training=True):
        raise NotImplementedError()

class IdentityDropoutSource(IdentitySource):
    def __init__(self, training=True):
        raise NotImplementedError()
