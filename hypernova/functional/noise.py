# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Noise
~~~~~
Noise sources.
"""
import math
import torch


class IIDSource(object):
    """
    Superclass for i.i.d. noise and dropout sources. Implements methods that
    toggle between test and train mode.

    There's nothing about this implementation level that requires the i.i.d.
    assumption.
    """
    def __init__(self, training):
        self.training = training

    def train(self, mode=True):
        self.training = mode

    def eval(self):
        self.train(False)


class IIDNoiseSource(IIDSource):
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
    def __init__(self, std=0.05, training=True):
        super(IIDNoiseSource, self).__init__(training)
        self.std = std

    def inject(self, tensor):
        if self.training:
            return tensor + self.sample(tensor.size()[:-1])
        else:
            return tensor


class IIDDropoutSource(IIDSource):
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
    def __init__(self, p=0.5, training=True):
        super(IIDDropoutSource, self).__init__(training)
        self.p = 1 - p

    def inject(self, tensor):
        if self.training:
            return tensor * self.sample(tensor.size()[:-1])
        else:
            return tensor


class DiagonalNoiseSource(IIDNoiseSource):
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

    Methods
    ----------
    eval
        Switch the source into inference mode.

    train
        Switch the source into training mode.

    sample(dim)
        Samples a random matrix that is zero everywhere except for a single
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

    inject(tensor)
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
    def __init__(self, std=0.05, offset=0, training=True):
        super(DiagonalNoiseSource, self).__init__(std, training)
        self.offset = offset

    def sample(self, dim):
        dim = [*dim[:-1], dim[-1] - abs(self.offset)]
        if self.training:
            noise = self.std * torch.randn(*dim)
            return torch.diag_embed(noise, self.offset)
        else:
            return 0


class SPSDNoiseSource(IIDNoiseSource):
    """
    Symmetric positive semidefinite noise source. Note that diagonal entries
    are effectively sampled from a very different distribution than
    off-diagonal entries. Be careful using this source; examine outputs for the
    dimension that you will be using before using it, as it exhibits some
    potentially very undesirable properties. This will be revisited in the
    future to determine if a better source can be provided.

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

    Methods
    ----------
    eval
        Switch the source into inference mode.

    train
        Switch the source into training mode.

    sample(dim)
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
            noise source.

    inject(tensor)
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
    def __init__(self, std=0.05, rank=None, training=True):
        super(SPSDNoiseSource, self).__init__(std, training)
        self.rank = rank

    def sample(self, dim):
        if self.training:
            rank = self.rank or dim[-1]
            noise = torch.empty((*dim, rank))
            var = self.std / math.sqrt(rank + (rank ** 2) / dim[-1])
            noise.normal_(std=math.sqrt(var))
            return noise @ noise.transpose(-1, -2)
        else:
            return 0


class DiagonalDropoutSource(IIDDropoutSource):
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

    Methods
    ----------
    eval
        Switch the source into inference mode.

    train
        Switch the source into training mode.

    sample(dim)
        Sample a random masking matrix from the dropout source.

        Parameters
        ----------
        dim : iterable
            Dimension of the matrices sampled from the source.

        Returns
        -------
        output : Tensor
            Block of masking matrices sampled from the noise source.

    inject(tensor)
        Apply dropout sampled from the source to an existing tensor block.

        Parameters
        ----------
        tensor : Tensor
            Tensor block into which to introduce the dropout sampled from the
            source.

        Returns
        -------
        output : Tensor
            Tensor block with dropout applied from the source.
    """
    def __init__(self, p=0.5, offset=0, training=True):
        super(DiagonalDropoutSource, self).__init__(p, training)
        self.offset = offset

    def sample(self, dim):
        dim = [*dim[:-1], dim[-1] - abs(self.offset)]
        if self.training:
            mask = (torch.rand(*dim) < self.p) / self.p
            return torch.diag_embed(mask, self.offset)
        else:
            return 1


class SPSDDropoutSource(IIDDropoutSource):
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
        Approximate probability of dropout across entries in each symmetric
        positive semidefinite masking matrix sampled from the source.
    rank : int or None (default None)
        Maximum rank of each masking matrix; inner dimension of the positive
        semidefinite product. Only a rank of 1 will intuitively correspond to
        standard dropout; any other rank might not yield a strictly on/off
        masking matrix.
    training : bool
        Indicates whether the source should operate under the assumption of
        training or inference; at test time, a noise-free sample is returned.

    Methods
    ----------
    eval
        Switch the source into inference mode.

    train
        Switch the source into training mode.

    sample(dim)
        Sample a random masking matrix from the dropout source.

        Parameters
        ----------
        dim : iterable
            Dimension of the matrices sampled from the source.

        Returns
        -------
        output : Tensor
            Block of masking matrices sampled from the noise source.

    inject(tensor)
        Apply dropout sampled from the source to an existing tensor block.

        Parameters
        ----------
        tensor : Tensor
            Tensor block into which to introduce the dropout sampled from the
            source.

        Returns
        -------
        output : Tensor
            Tensor block with dropout applied from the source.
    """
    def __init__(self, p=0.5, rank=1, training=True):
        super(SPSDDropoutSource, self).__init__(p, training)
        self.rank = rank

    def sample(self, dim):
        if self.training:
            rank = self.rank or dim[-1]
            mask = (torch.rand(*dim, rank) < self.p) / self.p
            print(mask)
            return mask @ mask.transpose(-1, -2) / rank
        else:
            return 1


class IdentitySource(IIDSource):
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
