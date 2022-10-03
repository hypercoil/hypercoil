# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Sylo ("symmetric low-rank") kernel operator.

This operation is a BrainNetCNN-like function equipped with an inductive bias
that should be favourable for learning on a set of unordered dense matrices,
and designed with analogy to convolutional layers. There are still a lot of
quirks to work out before it's usable.
"""
from __future__ import annotations
from typing import Callable, Literal, Optional

import jax

from ..engine import Tensor
from .crosssim import crosshair_similarity
from .matrix import delete_diagonal, expand_outer, sym2vec


# TODO: marking this as an experimental function (or add some tests)
def sylo(
    X: Tensor,
    L: Tensor,
    R: Optional[Tensor] = None,
    C: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    symmetry: Optional[Literal["cross", "skew"]] = None,
    similarity: Callable = crosshair_similarity,
    remove_diagonal: bool = False,
) -> Tensor:
    r"""
    Sylo transformation of a tensor block.

    Compute a local measure of graph or matrix similarity between an input
    and a bank of potentially symmetric, low-rank templates. In summary, the
    steps are:

    1. Outer product expansion of the left and right weight vectors into a bank
       of templates.
    2. Low-rank mapping of local similarities between the input and each of the
       templates.
    3. Biasing each filtered map.

    .. note::
        The forward operation of the ``sylo`` module does not preserve
        positive semidefiniteness, but it can be enforced by passing symmetric
        output through a transformation like the matrix exponential.

    :Dimension: **Input :** :math:`(N, *, C_{in}, H, W)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, :math:`C_{in}` denotes number of
                    input data channels, H and W denote height and width of
                    each input matrix.
                **L :** :math:`(*, C_{out}, C_{in}, H, rank)`
                    :math:`C_{out}` denotes number of output data channels,
                    and `rank` denotes the maximum rank of each template in
                    the reference bank.
                **R :** :math:`(*, C_{out}, C_{in}, W, rank)`
                    As above.
                **C :** :math:`(*, C_{out}, C_{in}, rank, rank)`
                    As above.
                **bias :** :math:`C_{out}`
                    As above.
                **Output :** :math:`(N, *, C_{out}, H, W)`
                    As above.

    Parameters
    ----------
    input : Tensor
        Input tensor of shape :math:`N \times C_{in} \times H \times W`.
    L, R : Tensors
        Left and right precursors of a low-rank basis that transforms the input
        via a local similarity measure. The template basis itself is created
        as the outer product between the left (column) and right (row) weight
        vectors. One way to enforce symmetry and positive semidefiniteness of
        learned templates is by passing the same tensor as `L` and `R`; this is
        the default behaviour.
    C : Tensor or None (default None)
        Coupling term. If this is specified, each template in the basis is
        modulated according to the coefficients in the coupling matrix.
        Providing a vector is equivalent to providing a diagonal coupling
        matrix. This term can, for instance, be used to toggle between
        positive and negative semidefinite templates.
    bias: Tensor
        Bias term to be added to the output.
    symmetry : ``'cross'``, ``'skew'``, or other (default None)
        Symmetry constraint imposed on the generated low-rank template matrix.

        * ``cross`` enforces symmetry by replacing the initial expansion with
          the average of the initial expansion and its transpose,
          :math:`\frac{1}{2} \left( L R^\intercal + R L^\intercal \right)`
        * ``skew`` enforces skew-symmetry by subtracting from the initial
          expansion its transpose,
          :math:`\frac{1}{2} \left( L R^\intercal - R L^\intercal \right)`
        * Otherwise, no explicit symmetry constraint is imposed. Symmetry can
          also be enforced by passing None for R or by passing the same input
          for R and L. (This approach also guarantees that the output is
          positive semidefinite.)
        This option will result in an error for nonsquare matrices or bipartite
        graphs. Note that the parameter count doubles if this is False.
    similarity : function (default `crosshair_similarity`)
        Definition of the similarity metric. This must be a function that takes
        as its first input the input tensor :math:`X` and as its second input
        the expanded (and potentially symmetrised) weight tensor. Similarity is
        computed between each of the N matrices in the first input stack and
        the weight. Uses the :func:`crosshair_similarity` measure by default.
    remove_diagonal : bool (default False)
        If True, the diagonal of the input matrix is removed before computing
        the similarity measure. This is useful for graphs, where the diagonal
        is often used to encode node attributes, or might be 1 for all nodes.

    Returns
    -------
    output : Tensor
        Input subject to a sylo transformation, as parametrised by the weights.
    """
    W = expand_outer(L, R, C=C, symmetry=symmetry)
    if remove_diagonal:
        # Deleting diagonals of both W and X is redundant/wasteful with all of
        # our current similarity measures.
        W = delete_diagonal(W)
        # X = delete_diagonal(X)
    output = similarity(X, W)
    if bias is not None:
        output += bias[..., None, None]
    if remove_diagonal:
        output = delete_diagonal(output)
    return output


def recombine(
    input: Tensor,
    mixture: Tensor,
    query: Optional[Tensor] = None,
    query_L: Optional[Tensor] = None,
    query_R: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
) -> Tensor:
    """
    Create a new mixture of the input feature maps.

    Parameters
    ----------
    input: Tensor ``(N x C_in x H x W)``
        Stack of input matrices or feature maps.
    mixture: Tensor ``(C_out x C_in)``
        Mixture matrix or recombinator.
    query: Tensor ``(N x C_in x C_in)``
        If provided, the mixture is recomputed as the dot product similarity
        between each mixture vector and each query vector, and the softmax of
        the result is used to form convex combinations of inputs.
    query_L: Tensor ``(C_out x C_in)`` and ``query_R: Tensor ``(C_out x C_in)``
        If provided, the mixture is recomputed as the dot product similarity
        between each mixture vector and a low-rank expansion of the left and
        right query vectors, and the softmax of the result is used to form
        convex combinations of inputs. Only one of ``query`` and ``query_L`` +
        ``query_R`` should be provided.
    bias: Tensor ``(C_in)``
        Bias term to apply after recombining.

    Returns
    -------
    output: Tensor ``(N x C_out x H x W)``
        Stack of output matrices or feature maps.
    """
    if query_L is not None:
        mixture = mixture @ query_L
        if query_R is None:
            query_R = query_L
        mixture = mixture @ query_R.swapaxes(-1, -2)
        mixture = jax.nn.softmax(mixture, -1)[..., None, :, :]
    if query is not None:
        mixture = mixture @ query
        mixture = jax.nn.softmax(mixture, -1)[..., None, :, :]
    output = (mixture @ input.swapaxes(-3, -2)).swapaxes(-3, -2)
    if bias is not None:
        output = output + bias[..., None, None]
    return output


def vertical_compression(
    input: Tensor,
    row_compressor: Tensor,
    col_compressor: Optional[Tensor] = None,
    renormalise: bool = True,
    remove_diagonal: bool = False,
    fold_channels: bool = False,
    sign: Optional[int] = None,
) -> Tensor:
    r"""
    Vertically compress a matrix or matrix stack of dimensions
    :math:`H_{in} \times W_{in} \rightarrow H_{out} \times W_{out}`.

    Parameters
    ----------
    input: Tensor
        Tensor to be compressed. This can be either a matrix of dimension
        :math:`H_{in} \times W_{in}` or a stack of such matrices, for
        instance of dimension :math:`N \times C \times H_{in} \times W_{in}`.
    row_compressor: Tensor
        Compressor for the rows of the input tensor. This should be a matrix
        of dimension :math:`H_{out} \times H_{in}`.
    col_compressor: Tensor or None
        Compressor for the columns of the input tensor. This should be a
        matrix of dimension :math:`W_{out} \times W_{in}`. If this is None,
        then symmetry is assumed: the column compressor and row compressor are
        the same.
    renormalise: bool
        If True, the output is renormalised to have standard deviation equal
        to that of the input. When ``remove_diagonal=True``, the standard
        deviation also ignores the diagonal elements.
    remove_diagonal: bool
        If True, the diagonal elements of the input are removed before and
        after compression.
    sign: int or None
        If not None, this should be either 1 or -1. If -1, the output is
        multiplied by -1.

    Returns
    -------
    output: Tensor
        Compressed tensor.
    """
    if remove_diagonal:
        input = delete_diagonal(input)
    input = input[..., None, :, :]
    if col_compressor is None:
        col_compressor = row_compressor
    compressed = (row_compressor @ input) @ col_compressor.swapaxes(-2, -1)
    if remove_diagonal:
        compressed = delete_diagonal(compressed)
    if sign is not None:
        compressed = sign * compressed
    if renormalise:
        # fmt: off
        if remove_diagonal:
            fac = (
                sym2vec(compressed).std(-1, keepdims=True)
                / sym2vec(input).std(-1, keepdims=True)
            )[..., None]
        else:
            fac = (
                compressed.std((-1, -2), keepdims=True)
                / input.std((-1, -2), keepdims=True)
            )
        compressed = compressed / fac
        # fmt: on
    if fold_channels:
        h, w = compressed.shape[-2:]
        if compressed.ndim > 4:
            n = compressed.shape[0]
            compressed = compressed.reshape((n, -1, h, w))
        else:
            compressed = compressed.reshape((-1, h, w))
    return compressed
