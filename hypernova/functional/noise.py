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
    def __init__(self, training):
        self.training = training

    def train(self):
        self.training = True

    def test(self):
        self.training = False


class IIDNoiseSource(IIDSource):
    def __init__(self, std=0.05, training=True):
        super(IIDNoiseSource, self).__init__(training)
        self.std = std

    def inject(self, tensor):
        if self.training:
            return tensor + self.sample(tensor.size()[:-1])
        else:
            return tensor


class IIDDropoutSource(IIDSource):
    def __init__(self, p=0.5, training=True):
        super(IIDDropoutSource, self).__init__(training)
        self.p = p

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
        returned.

    Methods
    ----------
    test
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
