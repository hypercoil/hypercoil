# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Vertical compression layer.
"""
from __future__ import annotations
from typing import Literal, Optional

import jax
import jax.numpy as jnp
import equinox as eqx

from ..engine import Tensor
from ..engine.paramutil import _to_jax_array
from ..functional.sylo import vertical_compression


def random_bipartite_lattice(
    lattice_order: int,
    in_features: int,
    out_features: int,
    *,
    key: 'jax.random.PRNGKey',
):
    """
    Generate a random biregular graph ('bipartite lattice').

    For many purposes, a more sophisticated initialisation might be desirable.
    As one example, see the maximum potential bipartite graph initialisation
    method.

    Parameters
    ----------
    lattice_order
        Lattice order: the number of edges is ``lattice_order`` times the
        least common multiple of ``in_features`` and ``out_features``, at
        most ``in_features * out_features``.
    in_features
        Number of input vertices.
    out_features
        Number of output vertices.
    key
        Pseudo-random number generator key.

    Returns
    -------
    Tensor
        Biadjacency matrix of shape ``(out_features, in_features)``.
    """
    num_edges = lattice_order * jnp.lcm(in_features, out_features)
    num_edges = min(num_edges, in_features * out_features)
    num_per_output = num_edges // out_features
    num_per_input = num_edges // in_features
    lattice = jnp.zeros((out_features, in_features))
    idx = jnp.arange(in_features)
    for i in range(out_features):
        remaining = num_per_input - lattice.sum(0)
        done = remaining <= 0
        num_done = done.sum()
        required = remaining >= (out_features - i)
        required_row = jnp.where(required)[0]
        num_required = len(required_row)
        sample_idx = idx[jnp.logical_and(~done, ~required)]
        key = jax.random.split(key, 1)[0]
        row = jax.random.choice(
            key,
            a=in_features - num_done - num_required,
            shape=(num_per_output - num_required,),
            replace=False,
        )
        row = jnp.concatenate((required_row, sample_idx[row]))
        lattice = lattice.at[i, row].set(1)
    return lattice


class VerticalCompression(eqx.Module):
    r"""
    Layer that compresses a graph by fusing vertices.

    For an adjacency matrix :math:`A`, the layer computes
    :math:`\left(C_{row} A\right) C_{col}^{\intercal}`, which maps an
    :math:`H_{in} \times W_{in}` matrix to an :math:`H_{out} \times W_{out}`
    matrix.

    Parameters
    ----------
    in_features
        Number of input vertices.
    out_features
        Number of output vertices.
    lattice_order
        Order of the random bipartite lattice that masks the weight, as in
        :func:`random_bipartite_lattice`.
    out_channels
        Number of output channels.
    renormalise
        Rescale the output to the standard deviation of the input, as in
        :func:`~hypercoil.functional.sylo.vertical_compression`.
    fold_channels
        Merge the channel axes of the output, as in
        :func:`~hypercoil.functional.sylo.vertical_compression`.
    forward_operation
        Operation of a call: ``'compress'``, ``'uncompress'`` or
        ``'reconstruct'``.
    sign
        Sign of the output: 1 or -1.
    key
        Pseudo-random number generator key.
    """
    in_features: int
    out_features: int
    out_channels: int
    sparsity: float
    _weight: Tensor
    mask: Tensor

    renormalise: bool
    fold_channels: bool
    sign: int
    forward_operation: Literal['compress', 'uncompress', 'reconstruct']

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lattice_order: int,
        out_channels: int = 1,
        renormalise: bool = True,
        fold_channels: bool = True,
        forward_operation: (
            Literal['compress', 'uncompress', 'reconstruct']
        ) = 'compress',
        sign: int = 1,
        *,
        key: 'jax.random.PRNGKey',
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.out_channels = out_channels
        self.renormalise = renormalise
        self.fold_channels = fold_channels
        self.forward_operation = forward_operation
        self.sign = sign

        key_m, key_w = jax.random.split(key, 2)
        mask = random_bipartite_lattice(
            lattice_order=lattice_order,
            in_features=in_features,
            out_features=out_features,
            key=key_m,
        )
        # TODO: work out a better default scale.
        weight = jax.random.uniform(
            key=key_w,
            shape=(out_channels, out_features, in_features),
        )

        self.sparsity = mask.sum() / mask.size
        self.mask = mask
        self._weight = weight

    @property
    def weight(self) -> Tensor:
        """Compression weight, zero outside the lattice mask."""
        mask, weight = _to_jax_array(self.mask), _to_jax_array(self._weight)
        return mask * weight

    @staticmethod
    def mode(
        model: 'VerticalCompression',
        mode: Literal['train', 'eval'],
    ) -> 'VerticalCompression':
        """Return ``model`` with ``forward_operation`` set to ``mode``."""
        return eqx.tree_at(
            lambda m: m.forward_operation,
            model,
            mode,
        )

    def uncompress(self, compressed: Tensor) -> Tensor:
        """Map a compressed matrix back to the input size."""
        return vertical_compression(
            input=compressed,
            row_compressor=self.weight.swapaxes(-1, -2),
            renormalise=self.renormalise,
            remove_diagonal=True,
            fold_channels=self.fold_channels,
            sign=self.sign,
        )

    def compress(self, input: Tensor) -> Tensor:
        """Compress ``input`` by fusing vertices."""
        return vertical_compression(
            input=input,
            row_compressor=self.weight,
            renormalise=self.renormalise,
            remove_diagonal=True,
            fold_channels=self.fold_channels,
            sign=self.sign,
        )

    def reconstruct(self, input: Tensor) -> Tensor:
        """Compress ``input``, then uncompress the result."""
        return self.uncompress(self.compress(input=input))

    def __call__(
        self,
        input: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        if self.forward_operation == 'compress':
            return self.compress(input=input)
        elif self.forward_operation == 'uncompress':
            return self.uncompress(compressed=input)
        elif self.forward_operation == 'reconstruct':
            return self.reconstruct(input=input)
