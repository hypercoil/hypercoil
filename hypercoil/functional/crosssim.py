# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Crosshair-kernel similarity
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Localised similarity functions over a crosshair kernel.
"""
import torch
from .crosshair import (
    crosshair_dot,
    crosshair_norm_l1,
    crosshair_norm_l2
)
from .matrix import expand_outer


def crosshair_similarity(X, W):
    """
    Crosshair kernel inner product between a tensor block and a reference block
    of template tensors.

    Note that each output slice of this procedure is often a matrix
    approximately of rank 1 (first singular value will dominate variance).

    Dimension
    ---------
    - X: :math:`(N, *, C_{in}, H, W)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      :math:`C_{in}` denotes number of input data channels, H and W denote
      height and width of each input matrix.
    - W: :math:`(*, C_{out}, C_{in}, H, W)`
      :math:`C_{out}` denotes number of output data channels.
    - Output: :math:`(N, *, C_{out}, H, W)`

    Parameters
    ----------
    X : Tensor
        Block of input tensors in which features should be identified. The last
        two axes correspond to the column and row dimensions of the matrices;
        the third to last axis corresponds to the data channels viewed by each
        reference template or filter. This is analogous to colour channels in a
        convolutional layer.
    W : Tensor
        Block of reference or template tensors or filters. The last two axes
        correspond to row and column dimensions, and the third to last
        corresponds to the input data channels. The fourth to last axis
        corresponds to output data channels.

    Output
    ------
    out : Tensor
        Output data channels resulting from applying the template to the input
        dataset.
    """
    return crosshair_dot(X.unsqueeze(-4), W).sum(-3)


def crosshair_cosine_similarity(X, W):
    """
    Crosshair kernel cosine similarity between a tensor block and a reference
    block of template tensors.

    Note that each output slice of this procedure is often a matrix
    approximately of rank 1 (first singular value will dominate variance).

    Dimension
    ---------
    - X: :math:`(N, *, C_{in}, H, W)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      :math:`C_{in}` denotes number of input data channels, H and W denote
      height and width of each input matrix.
    - W: :math:`(C_{out}, C_{in}, H, W)`
      :math:`C_{out}` denotes number of output data channels.
    - Output: :math:`(N, *, C_{out}, H, W)`

    Parameters
    ----------
    X : Tensor
        Block of input tensors in which features should be identified. The last
        two axes correspond to the column and row dimensions of the matrices;
        the third to last axis corresponds to the data channels viewed by each
        reference template or filter. This is analogous to colour channels in a
        convolutional layer.
    W : Tensor
        Block of reference or template tensors or filters. The last two axes
        correspond to row and column dimensions, and the third to last
        corresponds to the input data channels. The first axis corresponds to
        output data channels.

    Output
    ------
    out : Tensor
        Output data channels resulting from applying the template to the input
        dataset.
    """
    num = crosshair_dot(X.unsqueeze(-4), W)
    denom0 = crosshair_norm_l2(X.unsqueeze(-4))
    denom1 = crosshair_norm_l2(W)
    return (num / (denom0 * denom1)).sum(-3)


def crosshair_l1_similarity(X, W):
    """
    Crosshair kernel L1 distance between a tensor block and a reference block
    of template tensors.

    Note that each output slice of this procedure is often a matrix
    approximately of rank 1 (first singular value will dominate variance).

    Dimension
    ---------
    - X: :math:`(N, *, C_{in}, H, W)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      :math:`C_{in}` denotes number of input data channels, H and W denote
      height and width of each input matrix.
    - W: :math:`(C_{out}, C_{in}, H, W)`
      :math:`C_{out}` denotes number of output data channels.
    - Output: :math:`(N, *, C_{out}, H, W)`

    Parameters
    ----------
    X : Tensor
        Block of input tensors in which features should be identified. The last
        two axes correspond to the column and row dimensions of the matrices;
        the third to last axis corresponds to the data channels viewed by each
        reference template or filter. This is analogous to colour channels in a
        convolutional layer.
    W : Tensor
        Block of reference or template tensors or filters. The last two axes
        correspond to row and column dimensions, and the third to last
        corresponds to the input data channels. The first axis corresponds to
        output data channels.

    Output
    ------
    out : Tensor
        Output data channels resulting from applying the template to the input
        dataset.
    """
    diff = X.unsqueeze(-4) - W
    return crosshair_norm_l1(diff).sum(-3)


def crosshair_l2_similarity(X, W):
    """
    Crosshair kernel L2 distance between a tensor block and a reference block
    of template tensors.

    Note that each output slice of this procedure is often a matrix
    approximately of rank 1 (first singular value will dominate variance).

    Dimension
    ---------
    - X: :math:`(N, *, C_{in}, H, W)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      :math:`C_{in}` denotes number of input data channels, H and W denote
      height and width of each input matrix.
    - W: :math:`(C_{out}, C_{in}, H, W)`
      :math:`C_{out}` denotes number of output data channels.
    - Output: :math:`(N, *, C_{out}, H, W)`

    Parameters
    ----------
    X : Tensor
        Block of input tensors in which features should be identified. The last
        two axes correspond to the column and row dimensions of the matrices;
        the third to last axis corresponds to the data channels viewed by each
        reference template or filter. This is analogous to colour channels in a
        convolutional layer.
    W : Tensor
        Block of reference or template tensors or filters. The last two axes
        correspond to row and column dimensions, and the third to last
        corresponds to the input data channels. The first axis corresponds to
        output data channels.

    Output
    ------
    out : Tensor
        Output data channels resulting from applying the template to the input
        dataset.
    """
    diff = X.unsqueeze(-4) - W
    return crosshair_norm_l2(diff).sum(-3)
