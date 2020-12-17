# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
SPD Noise
~~~~~~~~~
Modules that inject symmetric positive (semi)definite noise.
"""
import torch
from torch.nn import Module, Parameter, init
from ..functional import SPSDNoiseSource
from ..functional.cov import corrnorm


class SPDNoise(Module):
    """
    Symmetric positive definite noise injection that preserves positive
    semidefiniteness.

    The input tensor is added together with noise sampled from a source that
    produces random symmetric positive semidefinite matrices. The result is
    thus guaranteed to remain in the positive semidefinite cone if the input is
    also positive semidefinite. The addition of noise is optionally followed by
    a renormalisation.

    Dimension
    ---------
    - Input: :math:`(*, N, N)`
      `*` denotes any number of preceding dimensions and N denotes the size of
      each square matrix.
    - Output: :math:`(*, N, N)`

    Parameters
    ----------
    std : nonnegative float
        Elementwise standard deviation of the sampled noise.
    norm : bool
        Indicates that the noisy sample should be renormalised after the
        injection of noise. The normalisation proceeds by computing the square
        root of each entry along the diagonal and embedding the reciprocals of
        the computed square roots into a new vector. The rank-1 outer product
        of this vector with itself is then multiplied elementwise with the
        noisy sample to normalise it. Because the outer product of a vector
        with itself is necessarily positive semidefinite, the Schur product
        theorem requires the result to be positive semidefinite.

    Attributes
    ----------
    noise : SPSDNoiseSource
        Symmetric positive semidefinite noise source.
    """
    def __init__(self, std, norm=True):
        super(SPDNoise, self).__init__()
        self.norm = norm
        self.noise = SPSDNoiseSource(std=std)

    def train(self, mode=True):
        super(_Cov, self).train(mode)
        self.noise.train(mode)

    def eval(self):
        super(_Cov, self).eval()
        self.noise.eval()

    def __repr__(self):
        return f'SPDNoise(std={self.noise.std}, norm={self.norm})'

    def forward(self, input):
        x = self.noise.inject(input)
        if self.norm:
            return x / corrnorm(x)
        else:
            return x
