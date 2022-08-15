# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities for batch-final, common-index COO sparse matrices.

We use the following convention for sparse matrices:
- Sparse matrices are represented using the BCOO format.
- All members of a batch are stored in a single COO matrix.
- All members of a batch are indexed by a common index. In other words, any
  entry that is non-zero in a batch member is also non-zero in all other batch
  members, or else must be encoded using an explicit zero entry. A consequence
  of this scheme is that the memory cost of representing a batch of matrices
  is greater relative to minimal sparse schemes. However, this scheme also
  benefits from the ability to vectorise operations on the batch.
- The leading dimensions of the matrix are the sparse dimensions.
- The final, dense dimension is the batch dimension.

It's possible -- very likely -- that we'll eventually figure out that this
isn't the best way to represent the kinds of sparse matrices that we work
with. If so, we'll want to change the interfaces in this module. As such,
any contents of this module should be considered experimental.
"""
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from typing import Sequence

from .utils import Tensor


def random_sparse(key, shape, density=0.1):
    """
    Generate a random sparse matrix.
    """
    n = jnp.prod(jnp.array(shape))
    nse = int(density * n)
    k1, k2 = jax.random.split(key)
    indices = jax.random.choice(k1, a=n, shape=(nse,), replace=False)
    indices = jnp.stack(jnp.unravel_index(indices, shape), axis=-1)
    data = jax.random.normal(k2, (nse,))
    return BCOO((data, indices), shape=shape).sum_duplicates()


def to_batch(matrices: Sequence[Tensor]) -> Tensor:
    """
    Convert a sequence of sparse matrices to a batch of matrices using the
    batch-final, common-index COO format.

    .. note::
        This function is not intended to be compatible with JIT compilation.
    """
    batch_size = len(matrices)
    shape = sum([m.data.shape[0] for m in matrices])
    remaining_shape = matrices[0].data.shape[1:]
    indices = jnp.concatenate([m.indices for m in matrices], axis=0)
    data = jnp.zeros((shape, *remaining_shape, batch_size))
    start = 0
    for i, matrix in enumerate(matrices):
        end = start + matrix.data.shape[0]
        data = data.at[start:end, ..., i].set(matrix.data)
        start = end
    return BCOO(
        (data, indices),
        shape=(*matrices[0].shape, batch_size)
    ).sum_duplicates()


def _get_dense_dim_mm(lhs, rhs):
    """
    Get the dense dimension of the matrix multiplication.
    """
    lhs_dims = lhs.data.shape[1:]
    rhs_dims = rhs.data.shape[1:]
    # we don't check for broadcastability here
    return [max(l, r) for l, r in zip(lhs_dims, rhs_dims)]


def spspmm(lhs, rhs, inner_dims=(0, 0), outer_dims=(1, 1)):
    """
    Sparse-sparse matrix multiplication with vectorisation over any dense
    dimensions.
    """
    # lhs_shape = lhs.shape
    # rhs_shape = rhs.shape
    # lhs_dims = list(lhs_shape[:-lhs.n_dense])
    # rhs_dims = list(rhs_shape[:-lhs.n_dense])
    # lhs_dims[outer_dims[0]] = lhs_dims[inner_dims[0]] = None
    # rhs_dims[outer_dims[0]] = rhs_dims[inner_dims[0]] = None

    # only support 2D sparse for now
    assert lhs.n_sparse == rhs.n_sparse == 2
    dense_dim_out = _get_dense_dim_mm(lhs, rhs)
    out_shape = (
        lhs.shape[outer_dims[0]],
        rhs.shape[outer_dims[1]],
        *dense_dim_out
    )

    out_nse = lhs.nse * rhs.nse # memory use scales as product of NSEs
    lhs_data = lhs.data[None, ...]
    rhs_data = rhs.data[:, None, ...]

    lhs_contract_dim, rhs_contract_dim = inner_dims
    lhs_contract_idx = lhs.indices[:, lhs_contract_dim][None, :]
    rhs_contract_idx = rhs.indices[:, rhs_contract_dim][:, None]
    out_nonzero = (lhs_contract_idx == rhs_contract_idx)
    extra_idx = [None] * len(dense_dim_out)
    out_nonzero = out_nonzero[tuple([...] + extra_idx)]
    out_data = jnp.where(out_nonzero, lhs_data * rhs_data, 0.)

    lhs_indices = jnp.ones_like(lhs.indices).at[:, -2].set(
        lhs.indices[:, outer_dims[0]])
    rhs_indices = jnp.ones_like(rhs.indices).at[:, -1].set(
        rhs.indices[:, outer_dims[1]])
    out_indices = (lhs_indices[None, ...] * rhs_indices[:, None, ...])

    out_indices = out_indices.reshape(out_nse, -1)
    out_data = out_data.reshape(out_nse, *dense_dim_out)
    return BCOO((out_data, out_indices), shape=out_shape)
