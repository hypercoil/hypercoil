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
- The final dimension is the batch dimension.

It's possible -- very likely -- that we'll eventually figure out that this
isn't the best way to represent sparse matrices. If so, we'll probably
want to change the interface of this module.
"""
import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from typing import Sequence

from .utils import Tensor


def random_sparse(key, shape, density=0.1, dtype=jnp.float32):
    """
    Generate a random sparse matrix.
    """
    n = jnp.prod(jnp.array(shape))
    nnz = int(density * n)
    k1, k2 = jax.random.split(key)
    indices = jax.random.choice(k1, a=n, shape=(nnz,), replace=False)
    indices = jnp.stack(jnp.unravel_index(indices, shape), axis=-1)
    data = jax.random.normal(k2, (nnz,))
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
