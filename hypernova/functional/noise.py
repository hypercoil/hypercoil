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


def diag_noise(dim, std=0.05, offset=0, training=True):
    dim = [*dim[:-1], dim[-1] - abs(offset)]
    if training:
        noise = std * torch.randn(*dim)
        return torch.diag_embed(noise, offset)
    else:
        return 0


def spsd_noise(dim, rank=None, std=0.05, training=True):
    """
    Symmetric positive semidefinite noise source.

    This function samples a random matrix :math:`K \in \mathbb{R}^{d \times r}`
    and computes the rank-r positive semidefinite product :math:`KK^\intercal`.
    For the outcome entries to have standard deviation near :math:`\sigma`,
    each entry in K (sampled i.i.d.) is distributed as

    :math:`\mathcal{N}\left(0, \frac{\sigma}{\sqrt{r + \frac{r^2}{d}}}\right)`

    The mean of this noise source is not exactly zero, but it trends toward
    zero as the dimension d increases. Note: revisit this later and work out
    what is going on mathematically. Here's a start:
    https://math.stackexchange.com/questions/101062/ ...
    is-the-product-of-two-gaussian-random-variables-also-a-gaussian

    Dimension
    ---------
    - Output :math:`(*, d, d)`
      `*` denotes any number of preceding dimensions, and d denotes the size of
      each symmetric positive semidefinite matrix sampled from the source.

    Parameters
    ----------
    dim : iterable
        Dimension of the matrices sampled from the source; outer dimension of
        the positive semidefinite product.
    rank : int or None
        Maximum rank of each output matrix; inner dimension of the positive
        semidefinite product. If this is less than `dim`, the sampled matrices
        will have some zero eigenvalues; if it is greater or equal, they will
        likely be positive definite. Regardless of the value of this parameter,
        the output rank cannot be greater than `dim`.
    std : float
        Approximate standard deviation across entries in each symmetric
        positive semidefinite matrix sampled from the source.

    Returns
    -------
    output : Tensor
        Block of symmetric positive semidefinite matrices sampled from the
        noise source.
    """
    if training:
        rank = rank or dim[-1]
        noise = torch.empty((*dim, rank))
        var = std / math.sqrt(rank + (rank ** 2) / dim[-1])
        noise.normal_(std=math.sqrt(var))
        return noise @ noise.transpose(-1, -2)
    else:
        return 0


def diag_dropout_mask(dim, p=0.5, training=True):
    if training:
        mask = (torch.rand(*dim) < p) / p
        return torch.diag_embed(mask)
    else:
        return 1


def spsd_dropout_mask(dim, p=0.5, rank=1, training=True):
    if training:
        mask = (torch.rand(*dim, rank) < p) / p
        return mask @ mask.T / rank
    else:
        return 1
