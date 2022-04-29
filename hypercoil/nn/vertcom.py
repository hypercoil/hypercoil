# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Vertical compression layer.
"""
import torch
from torch import nn
from torch.nn import init, Parameter
from ..init.mpbl import (
    maximum_potential_bipartite_lattice_autoselect_order,
    corr_criterion
)
from ..init.masked import sparse_masked_kaiming_uniform_


class VerticalCompression(nn.Module):
    r"""
    Compress a graph by fusing vertices. For an adjacency matrix A, this
    layer applies the transform
    :math:`\left(C_{row} A\right) \times C_{col}^{\intercal}` so that an
    :math:`H_{in} x W_{in}` matrix is mapped to an :math:`H_{out} x W_{out}`
    matrix.

    Parameters
    ----------
    in_features: tuple(int)
        Number of vertices in the input adjacency matrix. If this is a tuple,
        it corresponds to (`H_in`, `W_in`).
    out_features: tuple(int)
        Number of vertices in the output adjacency matrix. If this is a tuple,
        it corresponds to (`H_out`, `W_out`).
    nomask: bool
        Indicates that no sparsity constraint should be placed on the
        compression matrices and that all vertices of the original graph are
        allowed to load on all vertices of the compressed graph. If True, then
        any parameters used for mask computation (potentials, order, iters,
        temperature, random_init, attenuation) have no effect.
    symmetric: bool
        Indicates that the compression is symmetric, in which case
        `C_row` = `C_col`. If True, then `in_features` and `out_features`
        must correspond to a square matrix.
    init: dict
        Dictionary of parameters to pass to the sparse Kaiming initialisation
        function.
        Default: {'nonlinearity': 'linear'}

    Mask Parameters
    ---------------
    For interpretability, the vertical compression can be constrained such
    that each row/column in the compressed matrix is constituted only from a
    limited subset of rows/columns in the original matrix. This is achieved
    through elementwise multiplication of the compression matrix with a sparse
    mask.

    In the graph setting, this corresponds to a limited subset of vertices
    fusing into each new vertex.

    To help achieve a reasonable compression, we use a greedy Monte Carlo
    algorithm to estimate a heuristic called the "maximum potential bipartite
    lattice" (MPBL); the estimate serves as the mask for the compression
    matrix. The MPBL algorithm computes a mask that has a regular form such
    that each old vertex maps to exactly the same number of new vertices and
    each new vertex is constituted from exactly the same number of old
    vertices.

    potentials: tuple(Tensor), Tensor, or None
        Tensors that guide the MPBL algorithm. The first potential guides row
        compression and is of dimension H_in x H_in, while the second
        potential guides column compression and is of dimension W_in x W_in.
        For symmetric compression, only a single tensor should be provided.
        The potentials should be a proxy for the mutual information between
        each pair of rows/columns. The MPBL algorithm favours creating a new
        vertex out of a set of old vertices such that the edges connecting the
        selected old vertices have the largest possible potentials.
    order: int or tuple
        Sparsity of the mask produced by the MPBL algorithm. A single integer
        value sets sparsity to a particular level, where smaller integers
        correspond to a sparser matrix. A 2-tuple searches across a range of
        sparsity levels to find the one that maximises the overall potential;
        the tuple entries control the lower and upper bounds of the search.
    iters: int
        Number of iterations of the greedy algorithm to run. Only the result
        that maximises the final potential will be saved into the mask.
    temperature: float
        Softmax temperature for the MPBL algorithm. 0 deterministically fuses
        the maximum potential that has not already been fused (default).
        Infinity fuses randomly.
    random_init: bool
        Procedural initialisation. True begins from a random vertex (default).
        False begins from a vertex connected to the edge with the maximum
        overall potential.
    attenuation: int
        Attenuation factor for the potentials matrix. The potentials joining
        each old vertex set that is compressed into a single new vertex are
        divided by the attenuation factor. This helps to prevent redundancy in
        the compression.
    objective: Tensor or None
        Objective matrix for determining the quality of the sparse vertical
        compression masks: the mask set that is most closely able to recover
        the objective matrix is used.

    Attributes
    ----------
    mask_row and mask_col: Tensor
        Non-learnable left and right masks that enforce sparsity of the
        compressor matrices C_row and C_col.
    C_row and C_col: Tensor
        The matrices that map a matrix of dimension `H_in` x `W_in`
        to a matrix of dimension `H_out` x `W_out`.
        C_row has dimension `H_out` x `H_in`.
        C_col has dimension `W_out` x `W_in`.
    potentials: (Tensor, Tensor)
        Propagated version of the potentials matrices, which can be used as
        inputs for initialising downstream vertical compression masks.
    """
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, nomask=False, symmetric=True,
                 init=None, potentials=None, order=(1, 10), iters=100,
                 temperature=0, random_init=True, attenuation=2,
                 objective=None):
        super(VerticalCompression, self).__init__()
        self.H_in, self.W_in = in_features
        self.H_out, self.W_out = out_features
        if isinstance(order, int):
            order = (order, order)
        if init is None:
            init = {'nonlinearity': 'linear'}
        if symmetric:
            if (self.H_in != self.W_in
                or self.H_out != self.W_out):
                raise TypeError('in_features and out_features must be '
                                'square for symmetric matrix')
            if potentials is None:
                potentials = torch.ones(self.H_in, self.H_in)
        elif potentials is None:
            potentials = (
                torch.ones(self.H_in, self.H_in),
                torch.ones(self.W_in, self.W_in),
            )
        self.nomask = nomask
        self.symmetric = symmetric
        self.init = init
        self.potentials = potentials
        self.order = order
        self.iters = iters
        self.temperature = temperature
        self.random_init = random_init
        self.attenuation = attenuation
        self.objective = objective

        self.reset_parameters()

    def init_compressor(self, mask):
        compressor = mask.clone()
        sparse_kaiming_uniform_(compressor, mask=mask, **self.init)
        compressor *= mask
        return compressor

    def reset_parameters(self):
        if self.nomask:
            self.mask_row = Parameter(torch.ones(self.H_out, self.H_in))
            self.C_row = Parameter(self.init_compressor(self.mask_row))
            if self.symmetric:
                self.mask_col = self.mask_row
                self.C_col = self.C_row
            else:
                self.mask_col = Parameter(torch.ones(self.W_out, self.W_in))
                self.C_col = Parameter(self.init_compressor(self.mask_col))
            self.sparsity = (1.0, 1.0)
            self.mask_row.requires_grad = False
            self.mask_col.requires_grad = False
            return
        (masks, _, self.potentials, _, _
            ) = maximum_potential_bipartite_lattice_autoselect_order(
                potentials=self.potentials,
                n_out=self.H_out,
                order_min=self.order[0],
                order_max=self.order[1],
                iters=self.iters,
                temperature=self.temperature,
                random_init=self.random_init,
                attenuation=self.attenuation,
                inner_criterion=corr_criterion,
                outer_criterion=corr_criterion)
        if self.symmetric:
            self.mask_row = self.mask_col = Parameter(masks)
            self.C_row = self.C_col = Parameter(
                self.init_compressor(self.mask_row))
        else:
            self.mask_row = Parameter(masks[0])
            self.mask_col = Parameter(masks[1])
            self.C_row = Parameter(self.init_compressor(self.mask_row))
            self.C_col = Parameter(self.init_compressor(self.mask_col))
        self.mask_row.requires_grad = False
        self.mask_col.requires_grad = False
        self.sparsity = ((torch.sum(self.mask_row) /
                          torch.numel(self.mask_row)).item(),
                         (torch.sum(self.mask_col) /
                          torch.numel(self.mask_col)).item())

    def extra_repr(self):
        return ('in_features={}, out_features={}, sparsity=({:.4}, {:.4})'
        ).format(
            (self.H_in, self.W_in),
            (self.H_out, self.W_out),
            self.sparsity[0], self.sparsity[1])

    def forward(self, input):
        return vertical_compression(
            input=input,
            row_compressor=(self.mask_row * self.C_row),
            col_compressor=(self.mask_col * self.C_col))


##TODO: move to `functional` at some point.
def vertical_compression(input, row_compressor, col_compressor=None):
    """Vertically compress a matrix or matrix stack of dimensions
    `H_in` x `W_in` to `H_out` x `W_out`.

    Parameters
    ----------
    input: Tensor
        Tensor to be compressed. This can be either a matrix of dimension
        H_in x W_in or a stack of such matrices, for instance of dimension
        N x C x H_in x W_in.
    row_compressor: Tensor
        Compressor for the rows of the input tensor. This should be a matrix
        of dimension H_out x H_in.
    col_compressor: Tensor or None
        Compressor for the columns of the input tensor. This should be a
        matrix of dimension W_out x W_in. If this is None, then symmetry is
        assumed: the column compressor and row compressor are the same.
    """
    if col_compressor is None:
        col_compressor = row_compressor
    return (row_compressor @ input) @ col_compressor.t()
