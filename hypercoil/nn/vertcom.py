# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Vertical compression layer.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Literal, Optional
from ..engine import Tensor
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
    """
    num_edges = lattice_order * jnp.lcm(in_features, out_features)
    num_edges = min(num_edges, in_features * out_features)
    num_per_output = num_edges // out_features
    num_per_input = num_edges // in_features
    lattice = jnp.zeros((out_features, in_features))
    idx = jnp.arange(in_features)
    for i in range(out_features):
        remaining = num_per_input - lattice.sum(0)
        done = (remaining <= 0)
        num_done = done.sum()
        required = (remaining >= (out_features - i))
        required_row = jnp.where(required)[0]
        num_required = len(required_row)
        sample_idx = idx[jnp.logical_and(~done, ~required)]
        key = jax.random.split(key, 1)[0]
        row = jax.random.choice(
            key,
            a=in_features - num_done - num_required,
            shape=(num_per_output - num_required,),
            replace=False
        )
        row = jnp.concatenate((required_row, sample_idx[row]))
        lattice = lattice.at[i, row].set(1)
    return lattice


class VerticalCompression(eqx.Module):
    r"""
    Compress a graph by fusing vertices. For an adjacency matrix A, this
    layer applies the transform
    :math:`\left(C_{row} A\right) \times C_{col}^{\intercal}` so that an
    :math:`H_{in} \times W_{in}` matrix is mapped to an
    :math:`H_{out} \times W_{out}` matrix.
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
            Literal['compress', 'uncompress', 'reconstruct']) = 'compress',
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
        #TODO: work out a better default scale.
        weight = jax.random.uniform(
            key=key_w,
            shape=(out_channels, out_features, in_features),
        )

        self.sparsity = mask.sum() / mask.size
        self.mask = mask
        self._weight = weight

    @property
    def weight(self) -> Tensor:
        return self.mask * self._weight

    @staticmethod
    def mode(
        model: 'VerticalCompression',
        mode: Literal['train', 'eval']
    ) -> 'VerticalCompression':
        return eqx.tree_at(
            lambda m: m.forward_operation,
            model,
            mode,
        )

    def uncompress(self, compressed: Tensor) -> Tensor:
        return vertical_compression(
            input=compressed,
            row_compressor=self.weight.swapaxes(-1, -2),
            renormalise=self.renormalise,
            remove_diagonal=True,
            fold_channels=self.fold_channels,
            sign=self.sign
        )

    def compress(self, input: Tensor) -> Tensor:
        return vertical_compression(
            input=input,
            row_compressor=(self.mask * self.weight),
            renormalise=self.renormalise,
            remove_diagonal=True,
            fold_channels=self.fold_channels,
            sign=self.sign
        )

    def reconstruct(self, input: Tensor) -> Tensor:
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
