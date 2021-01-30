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
from .matrix import toeplitz


class _IIDSource(torch.nn.Module):
    """
    Superclass for i.i.d. noise and dropout sources. Implements methods that
    toggle between test and train mode.

    There's nothing about this implementation level that requires the i.i.d.
    assumption.
    """
    def __init__(self, distr, training):
        super(_IIDSource, self).__init__()
        self.distr = distr
        self.training = training

    def train(self, mode=True):
        """
        Switch the source into training mode.
        """
        self.training = mode

    def eval(self):
        """
        Switch the source into inference mode.
        """
        self.train(False)

    def forward(self, input):
        """
        Inject noise sampled from the source into a tensor.
        """
        return self.inject(input)


class _IIDNoiseSource(_IIDSource):
    """
    Superclass for i.i.d. noise sources. Implements a method for injecting
    sampled noise additively into an existing tensor.

    Subclasses are responsible for implementing the correct `sample` method
    that accepts a dimension argument.

    See also
    --------
    IIDDropoutSource : For multiplicative sample injection.
    """
    def __init__(self, distr=None, training=True):
        distr = distr or torch.distributions.normal.Normal(
            torch.Tensor([0]), torch.Tensor([1]))
        super(_IIDNoiseSource, self).__init__(distr, training)

    def inject(self, tensor):
        """
        Inject noise sampled from the source into an existing tensor block.

        Parameters
        ----------
        tensor : Tensor
            Tensor block into which to introduce the noise sampled from the
            source.

        Returns
        -------
        output : Tensor
            Tensor block with noise injected from the source.
        """
        if self.training:
            return tensor + self.sample(tensor.size())
        else:
            return tensor

    def extra_repr(self):
        return f'{self.distr}'


class _IIDSquareNoiseSource(_IIDNoiseSource):
    """
    Superclass for i.i.d. noise sources. Implements a method for injecting
    sampled noise additively into an existing tensor.

    Subclasses are responsible for implementing the correct `sample` method
    that accepts a dimension argument. Currently there is a square matrix
    input assumption in the `inject` method that future work might revise.

    See also
    --------
    IIDDropoutSource : For multiplicative sample injection.
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
            return tensor + self.sample(tensor.size()[:-1])
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
    IIDNoiseSource : For additive sample injection.
    """
    def __init__(self, distr=None, training=True):
        distr = distr or torch.distributions.bernoulli.Bernoulli(
            torch.Tensor([0.5]))
        super(_IIDDropoutSource, self).__init__(distr, training)

    def inject(self, tensor):
        if self.training:
            return tensor * self.sample(tensor.size())
        else:
            return tensor

    def extra_repr(self):
        return f'{self.distr}'


class _IIDSquareDropoutSource(_IIDDropoutSource):
    """
    Superclass for i.i.d. noise sources. Implements a method for injecting
    sampled noise multiplicatively into an existing tensor.

    Subclasses are responsible for implementing the correct `sample` method
    that accepts a dimension argument. Currently there is a square matrix
    input assumption in the `inject` method that future work might revise.

    See also
    --------
    IIDNoiseSource : For additive sample injection.
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
            return tensor * self.sample(tensor.size()[:-1])
        else:
            return tensor


class _AxialSampler(_IIDSource):
    def __init__(self, distr=None, training=True, sample_axes=None):
        self.sample_axes = sample_axes
        super(_AxialSampler, self).__init__(distr, training)

    def select_dim(self, dim):
        if self.sample_axes is not None:
            dim = list(dim)
            n_axes = len(dim)
            for ax in range(n_axes):
                if not (ax in self.sample_axes or
                    (ax - n_axes) in self.sample_axes):
                    dim[ax] = 1
        return dim


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
    def sample(self, dim):
        """
        Sample a random tensor of the specified shape, in which the entries
        are sampled i.i.d. from the specified distribution.

        Parameters
        ----------
        dim : iterable
            Dimension of the tensors sampled from the source.

        Returns
        -------
        output : Tensor
            Tensor sampled from the noise source.
        """
        dim = self.select_dim(dim)
        return self.distr.sample(dim).squeeze(-1)


class DiagonalNoiseSource(_IIDSquareNoiseSource):
    """
    Zero-mean diagonal noise source.

    Parameters/Attributes
    ---------------------
    std : float (default 0.05)
        Standard deviation of the noise source.
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
        diagonal band, along which the entries are sampled i.i.d. from a
        zero-mean Gaussian distribution with the specified standard deviation.

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

    Parameters/Attributes
    ---------------------
    std : float (default 0.05)
        Approximate standard deviation across entries in each symmetric
        positive semidefinite matrix sampled from the source.
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
    def __init__(self, distr=None, rank=None, training=True):
        super(LowRankNoiseSource, self).__init__(distr, training)
        self.rank = rank

    def sample(self, dim):
        """
        Sample a random matrix :math:`K \in \mathbb{R}^{d \times r}` and
        computes the rank-r positive semidefinite product :math:`KK^\intercal`.
        For the outcome entries to have standard deviation near :math:`\sigma`,
        each entry in K (sampled i.i.d.) is distributed as

        :math:`\mathcal{N}\left(0, \frac{\sigma}{\sqrt{r + \frac{r^2}{d}}}\right)`

        The mean of this noise source is not exactly zero, but it trends toward
        zero as the dimension d increases. Note: revisit this later and work
        out what is going on mathematically. Here's a start:
        https://math.stackexchange.com/questions/101062/ ...
        is-the-product-of-two-gaussian-random-variables-also-a-gaussian

        Parameters
        ----------
        dim : iterable
            Dimension of the matrices sampled from the source.

        Returns
        -------
        output : Tensor
            Block of symmetric, positive semidefinite matrices sampled from the
            low-rank noise source.
        """
        if self.training:
            rank = self.rank or dim[-1]
            noise = torch.empty((*dim, rank))
            var = self.distr.scale / math.sqrt(rank + (rank ** 2) / dim[-1])
            noise.normal_(std=math.sqrt(var))
            return noise @ noise.transpose(-1, -2)
        else:
            return 0


class SPSDNoiseSource(_IIDNoiseSource):
    """
    Symmetric positive semidefinite noise source.

    Uses the matrix exponential to project a symmetric noise matrix into the
    positive semidefinite cone. The symmetric matrix is diagonalised, its
    eigenvalues are exponentiated thereby ensuring each is positive, and it
    is recomposed. Note that due to numerical errors some extremely small
    negative eigenvalues can occur in the sampled matrix.

    Parameters/Attributes
    ---------------------
    std : float (default 0.05)
        Approximate standard deviation across entries in each symmetric
        positive semidefinite matrix sampled from the source.
    training : bool
        Indicates whether the source should operate under the assumption of
        training or inference; at test time, a noise-free sample is returned.
    """
    def __init__(self, distr=None, training=True):
        super(SPSDNoiseSource, self).__init__(distr, training)

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
            noise = torch.empty((*dim, dim[-1]))
            noise.normal_()
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
    def sample(self, dim):
        """
        Sample a random tensor of the specified shape, in which the entries
        are sampled i.i.d. from the specified distribution.

        Parameters
        ----------
        dim : iterable
            Dimension of the tensors sampled from the source.

        Returns
        -------
        output : Tensor
            Tensor sampled from the noise source.
        """
        dim = self.select_dim(dim)
        return self.distr.sample(dim).squeeze(-1) / self.distr.mean


class DiagonalDropoutSource(_IIDSquareDropoutSource):
    """
    Diagonal dropout source.

    Parameters/Attributes
    ---------------------
    p : float (default 0.5)
        Probability of dropout across entries along the indicated diagonal of
        each masking matrix sampled from the source.
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

    Parameters/Attributes
    ---------------------
    p : float (default 0.5)
        Approximate probability of dropout across columns in each symmetric
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
            mask = self.distr.sample((*dim, 1)).squeeze(-1)
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

    Parameters/Attributes
    ---------------------
    p : float (default 0.5)
        Approximate probability of dropout across columns in each symmetric
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
    """
    A source that does nothing at all.
    """
    def __init__(self, training=True):
        super(IdentitySource, self).__init__(training)

    def inject(self, tensor):
        return tensor


class IdentityNoiseSource(IdentitySource):
    """
    A source that does nothing at all, implemented with additive identity.
    """
    def __init__(self, training=True):
        super(IdentityNoiseSource, self).__init__(training)
        self.std = 0

    def sample(self, dim):
        return 0


class IdentityDropoutSource(IdentitySource):
    """
    A source that does nothing at all, implemented with multiplicative
    identity.
    """
    def __init__(self, training=True):
        super(IdentityDropoutSource, self).__init__(training)
        self.p = 1

    def sample(self, dim):
        return 1
