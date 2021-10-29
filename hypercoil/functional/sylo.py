# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Sylo
~~~~
Sylo ("symmetric low-rank") kernel operator.
"""
from .matrix import expand_outer
from .crosssim import crosshair_similarity


def sylo(X, L, R=None, bias=None, symmetry=None,
         similarity=crosshair_similarity):
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

    Returns
    -------
    output : Tensor
        Input subject to a sylo transformation, as parametrised by the weights.
    """
    W = expand_outer(L, R, symmetry)
    output = similarity(X, W)
    if bias is not None:
        output += bias[..., None, None]
    return output
