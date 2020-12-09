# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Sylo
~~~~
Sylo ("symmetric low-rank") kernel operator.
"""
from .functions.crosssim import crosshair_similarity


def sylo(input, weight, bias=None, symmetry=None,
         similarity=crosshair_similarity):
    """
    Compute a local measure of graph or matrix similarity between an input
    and a bank of potentially symmetric, low-rank templates.
    In summary, the steps are:
    0. Shape: N x C_in x H x W
    1. Low-rank mapping of local similarities between the input and each of
       the weights
       Shape: N x C_out x H x W
    2. Biasing each filtered map
       Shape: N x C_out x H x W
    Parameters
    ----------
    input: Tensor
        Input tensor of shape `N` x `C_in` x `H` x `W`.
    weight: tuple(Tensor)
        Low-rank basis that transforms the input via a local similarity
        measure. Passed as (`L`, `R`), where `L` is of shape
        `C_out` x `C_in` x `H` x `rank` and `R` is of shape
        `C_out` x `C_in` x `W` x `rank`. One way to enforce symmetry of
        learned templates is by passing the same tensor as `L` and `R`.
    bias: Tensor
        Bias term of shape `C_out` to be added to the output.
    symmetry: bool, 'cross', or 'skew'
        Symmetry constraints to impose on expanded templates.
        * If 'cross', the templates generated from the expansion of `weights`
          are constrained to be symmetric; the templates are defined as the
          average of the expansion and its transpose:
          1/2 (L @ R.T + R @ L.T).
        * If 'skew', the templates generated from the expansion of `weights`
          are constrained to be skew-symmetric; the templates are defined as
          the difference between the expansion and its transpose:
          L @ R.T - R @ L.T
        * Otherwise, no explicit symmetry constraints are placed on the
          templates generated from the expansion of `weights`. Symmetry can
          still be enforced by passing the same tensor twice for `weights`.
        This option will result in an error for nonsquare matrices or bipartite
        graphs. Note that the parameter count doubles if this is False.
        Default: None
    similarity: function
        Definition of the similarity metric. This must be a function whose
        inputs and outputs are:
        * input 0: reference matrix (N x C_in x H x W)
        * input 1: left template generator (C_out x C_in x H x R)
        * input 2: right template generator (C_out x C_in x H x R)
        * input 3: symmetry constraint ('cross', 'skew', or other)
        * output (N x C_out x H x W)
        Similarity is computed between each of the N matrices in the first
        input stack and the (low-rank) matrix derived from the outer-product
        expansion of the second and third inputs.
        Default: `crosshair_similarity`
    """
    L, R = weight
    output = similarity(input, L, R, symmetry)
    if bias is not None:
        output += bias.view(1, -1, 1, 1)
    return output
