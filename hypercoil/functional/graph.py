# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Measures on graphs and networks.
"""
import jax
import jax.numpy as jnp
from functools import partial, singledispatch
from jax.nn import relu
from jax.experimental.sparse import BCOO
from typing import Any, Callable, Literal, Optional, Union
from .sparse import TopKTensor, dspdmm, topk_diagaugment, topk_diagzero
from .matrix import delete_diagonal, fill_diagonal
from .utils import is_sparse
from ..engine import Tensor, vmap_over_outer


def girvan_newman_null(A: Tensor) -> Tensor:
    r"""
    Girvan-Newman null model for a tensor block.

    The Girvan-Newman null model is defined as the expected connection weight
    between each pair of vertices if all edges are cut and the resulting stubs
    then randomly rewired. For the vector of node in-degrees
    :math:`k_i \in \mathbb{R}^I`, vector of node out-degrees
    :math:`k_o \in \mathbb{R}^O`, and total edge weight
    :math:`2m \in \mathbb{R}`, this yields the null model

    :math:`P_{GN} = \frac{1}{2m} k_i k_o^\intercal`

    or, in terms of the adjacency matrix :math:`A \in \mathbb{R}^{I \times O}`

    :math:`P_{GN} = \frac{1}{\mathbf{1}^\intercal A \mathbf{1}} A \mathbf{1} \mathbf{1}^\intercal A`

    :Dimension: **Input :** :math:`(N, *, I, O)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, I denotes number of vertices in
                    the source set and O denotes number of vertices in the
                    sink set. If the same set of vertices emits and receives
                    edges, then :math:`I = O`.
                **Output :** :math:`(N, *, I, O)`
                    As above.

    Parameters
    ----------
    A : Tensor
        Block of adjacency matrices for which the Girvan-Newman null model is
        to be computed.

    Returns
    -------
    P : Tensor
        Block comprising Girvan-Newman null matrices corresponding to each
        input adjacency matrix.
    """
    k_i = A.sum(-1, keepdims=True)
    k_o = A.sum(-2, keepdims=True)
    two_m = k_i.sum(-2, keepdims=True)
    return k_i @ k_o / two_m


def modularity_matrix(
    A: Tensor,
    gamma: float = 1,
    null: Callable = girvan_newman_null,
    normalise: bool = False,
    sign: Optional[Literal['+', '-']] = '+',
    **params
):
    r"""
    Modularity matrices for a tensor block.

    The modularity matrix is defined as a normalised, weighted difference
    between the adjacency matrix and a suitable null model. For a weight
    :math:`\gamma`, an adjacency matrix :math:`A`, a null model :math:`P`, and
    total edge weight :math:`2m`, the modularity matrix is computed as

    :math:`B = \frac{1}{2m} \left( A - \gamma P \right)`

    :Dimension: **Input :** :math:`(N, *, I, O)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, I denotes number of vertices in
                    the source set and O denotes number of vertices in the
                    sink set. If the same set of vertices emits and receives
                    edges, then :math:`I = O`.
                **Output :** :math:`(N, *, I, O)`
                    As above.

    Parameters
    ----------
    A : Tensor
        Block of adjacency matrices for which the modularity matrix is to be
        computed.
    gamma : nonnegative float (default 1)
        Resolution parameter for the modularity matrix. A smaller value assigns
        maximum modularity to partitions with large communities, while a larger
        value assigns maximum modularity to partitions with many small
        communities.
    null : callable(A) (default ``girvan_newman_null``)
        Function of ``A`` that returns, for each adjacency matrix in the input
        tensor block, a suitable null model. By default, the
        :doc:`Girvan-Newman null model <hypercoil.functional.graph.girvan_newman_null>`
        is used.
    normalise : bool (default False)
        Indicates that the resulting matrix should be normalised by the total
        matrix degree. This may not be necessary for many use cases -- for
        instance, where the arg max of a function of the modularity matrix is
        desired.
    sign : ``'+'``, ``'-'``, or None (default ``'+'``)
        Sign of connections to be considered in the modularity.
    **params
        Any additional parameters are passed to the null model.

    Returns
    -------
    P : Tensor
        Block comprising modularity matrices corresponding to each input
        adjacency matrix.

    See also
    --------
    relaxed_modularity: Compute the modularity given a community structure.
    """
    if sign == '+':
        A = relu(A)
    elif sign == '-':
        A = -relu(-A)
    mod = A - gamma * null(A, **params)
    if normalise:
        two_m = A.sum((-2, -1), keepdims=True)
        return mod / two_m
    return mod


def coaffiliation(
    C_i: Tensor,
    C_o: Optional[Tensor] = None,
    L: Optional[Tensor] = None,
    exclude_diag: bool = True,
    normalise: bool = False
) -> Tensor:
    r"""
    Coaffiliation of vertices under a community structure.

    Given community affiliation matrices
    :math:`C^{(i)} \in \mathbb{R}^{I \times C}` for source nodes and
    :math:`C^{(o)} \in \mathbb{R}^{O \times C}` for sink nodes, and given a
    matrix of inter-community coupling coefficients
    :math:`\Omega \in \mathbb{R}^{C \times C}`, the coaffiliation
    :math:`H \in \mathbb{R}^{I \times O}` is computed as

    :math:`H = C^{(i)} \Omega C^{(o)\intercal}`

    :Dimension: **C_i :** :math:`(*, I, C)`
                    `*` denotes any number of preceding dimensions, I denotes
                    number of vertices in the source set, and C denotes the
                    total number of communities in the proposed partition.
                **C_o :** :math:`(*, I, C)`
                    O denotes number of vertices in the sink set. If the same
                    set of vertices emits and receives edges, then
                    :math:`I = O`.
                **L :** :math:`(*, C, C)`
                    As above.
                **Output :** :math:`(*, I, O)`
                    As above.

    Parameters
    ----------
    C_i : Tensor
        Community affiliation of vertices in the source set. Each slice is a
        matrix :math:`C^{(i)} \in \mathbb{R}^{I \ times C}` that encodes the
        uncertainty in each vertex's community assignment. :math:`C^{(i)}_{jk}`
        denotes the probability that vertex j is assigned to community k. If
        this is binary-valued, then it reflects a deterministic assignment.
    C_o : Tensor or None (default None)
        Community affiliation of vertices in the sink set. If None, then it is
        assumed that the source and sink sets are the same, and ``C_o`` is set
        equal to ``C_i``.
    L : Tensor or None (default None)
        The inter-community coupling matrix :math:`\Omega`, mapping the
        probability of affiliation between communities. Each entry
        :math:`L_{ij}` encodes the probability of a vertex in community i
        connecting with a vertex in community j. If None, then a strictly
        assortative structure is assumed (equivalent to L equals identity),
        under which nodes in the same community preferentially coaffiliate
        while nodes in different communities remain disaffiliated.
    exclude_diag : bool (default True)
        Indicates that self-links are not factored into the coaffiliation.
    normalise : bool (default False)
        Normalise all community assignment weights to max out at 1.

    Returns
    -------
    C : Tensor
        Coaffiliation matrix for each input community structure.
    """
    if C_o is None: C_o = C_i
    if normalise:
        norm_fac_i = jnp.maximum(1, C_i.max((-1, -2)))
        norm_fac_o = jnp.maximum(1, C_o.max((-1, -2)))
        C_i = C_i / norm_fac_i
        C_o = C_o / norm_fac_o
    if L is None:
        C = C_i @ C_o.swapaxes(-1, -2)
    else:
        C = C_i @ L @ C_o.swapaxes(-1, -2)
    if exclude_diag:
        C = delete_diagonal(C)
    return C


def relaxed_modularity(
    A: Tensor,
    C: Tensor,
    C_o: Optional[Tensor] = None,
    L: Optional[Tensor] = None,
    exclude_diag: bool = True,
    gamma: float = 1,
    null: Callable = girvan_newman_null,
    normalise_modularity: bool = True,
    normalise_coaffiliation: bool = True,
    directed: bool = False,
    sign: Optional[Literal['+', '-']] ='+',
    **params
) -> Tensor:
    r"""
    A relaxation of the modularity of a network given a community partition.

    This relaxation supports non-deterministic assignments of vertices to
    communities and non-assortative linkages between communities. It reverts
    to standard behaviour when the inputs it is provided are standard.

    The relaxed modularity is defined as the sum of all entries in the
    Hadamard (elementwise) product between the modularity matrix and the
    coaffiliation matrix.

    :math:`Q = \mathbf{1}^\intercal \left( B \circ H \right) \mathbf{1}`

    :Dimension: **Input :** :math:`(N, *, I, O)`
                    N denotes batch size, ``*`` denotes any number of
                    intervening dimensions, I denotes number of vertices in
                    the source set and O denotes number of vertices in the
                    sink set. If the same set of vertices emits and receives
                    edges, then :math:`I = O`.
                **C :** :math:`(*, I, C)`
                    C denotes the total number of communities in the proposed
                    partition.
                **C_o :** :math:`(*, I, C)`
                    As above.
                **L :** :math:`(*, C, C)`
                    As above.
                **Output :** :math:`(N, *)`
                    As above.

    Parameters
    ----------
    A : Tensor
        Block of adjacency matrices for which the modularity is to be computed.
    C : Tensor
        Community affiliation of vertices in the source set. Each slice is a
        matrix :math:`C^{(i)} \in \mathbb{R}^{I \ times C}` that encodes the
        uncertainty in each vertex's community assignment. :math:`C^{(i)}_{jk}`
        denotes the probability that vertex j is assigned to community k. If
        this is binary-valued, then it reflects a deterministic assignment.
    C_o : Tensor or None (default None)
        Community affiliation of vertices in the sink set. If None, then it is
        assumed that the source and sink sets are the same, and ``C_o`` is set
        equal to ``C``.
    L : Tensor or None (default None)
        Probability of affiliation between communities. Each entry
        :math:`L_{ij}` encodes the probability of a vertex in community i
        connecting with a vertex in community j. If None, then a strictly
        assortative structure is assumed (equivalent to L equals identity),
        under which nodes in the same community preferentially coaffiliate
        while nodes in different communities remain disaffiliated.
    exclude_diag : bool (default True)
        Indicates that self-links are not factored into the coaffiliation.
    gamma : nonnegative float (default 1)
        Resolution parameter for the modularity matrix. A smaller value
        assigns maximum modularity to partitions with large communities, while
        a larger value assigns maximum modularity to partitions with many
        small communities.
    null : callable(A) (default ``girvan_newman_null``)
        Function of ``A`` that returns, for each adjacency matrix in the input
        tensor block, a suitable null model. By default, the
        :doc:`Girvan-Newman null model <hypercoil.functional.graph.girvan_newman_null>`
        is used.
    normalise_modularity : bool (default True)
        Indicates that the resulting matrix should be normalised by the total
        matrix degree. This may not be necessary for many use cases -- for
        instance, where the arg max of a function of the modularity matrix is
        desired.
    normalise_coaffiliation : bool (default True)
        Indicates that all weights in the community assignment matrix block
        should be renormalised to max out at 1. Note that this is unnecessary
        if the affiliations have already been passed through a softmax.
    directed : bool (default False)
        Indicates that the input adjacency matrices should be considered as a
        directed graph.
    sign : ``'+'``, ``'-'``, or None (default ``'+'``)
        Sign of connections to be considered in the modularity.
    **params
        Any additional parameters are passed to the null model.

    Returns
    -------
    Q : Tensor
        Modularity of each input adjacency matrix.
    """
    B = modularity_matrix(A, gamma=gamma, null=null,
                          normalise=normalise_modularity, sign=sign, **params)
    C = coaffiliation(C, C_o=C_o, L=L, exclude_diag=exclude_diag,
                      normalise=normalise_coaffiliation)
    Q = (B * C).sum((-2, -1))
    if not directed:
        return Q / 2
    return Q


def degree(W: Tensor) -> Tensor:
    # TODO: technically, we *have* implemented this for sparse graphs in the
    # ``_sparse_laplacian`` function, but it involves some overhead that makes
    # a separate implementation useless for our purposes.
    if is_sparse(W):
        raise NotImplementedError
    return W.sum(-1)


@singledispatch
def graph_laplacian(
    W: Tensor,
    normalise: bool = True,
    topk: bool = True,
) -> Tensor:
    r"""
    Laplacian of a graph.

    Given the diagonal matrix of matrix degrees :math:`D`, the Laplacian
    :math:`L` of a graph with adjacency matrix :math:`A` is

    :math:`L = D - A`

    For many applications, vertices with large degrees tend to dominate
    properties of the Laplacian, and it is desirable to normalise the
    Laplacian before further analysis.

    .. math::

        \widetilde{L} = D^{-\frac{1}{2}} L D^{-\frac{1}{2}}

        \widetilde{L} = I - D^{-\frac{1}{2}} A D^{-\frac{1}{2}}

    .. note::

        For an undirected graph, each edge should be duplicated in the index
        and weight tensors, with source and target vertices swapped in the
        index.

    .. note::

        For a directed graph, this computes the row Laplacian. This could be
        either the in-degree Laplacian or the out-degree Laplacian, depending
        on the convention adopted for the input.

    :Dimension: **W :** :math:`(*, N, N)` or :math:`(*, E)`
                    ``*`` denotes any number of preceding dimensions, N
                    denotes number of vertices, and E denotes number of edges.
                    The shape should be :math:`(*, N, N)` if ``edge_index`` is
                    not provided and :math:`(*, E)` if ``edge_index`` is
                    provided.
                **edge_index :** :math:`(*, 2, E)`
                    As above.
                **Output :** :math:`(*, N, N)` or tuple(:math:`(*, E)`, :math:`(*, 2, E)`)
                    As above.

    Parameters
    ----------
    W : tensor
        Edge weight tensor. If ``edge_index`` is not provided, then this
        should be the graph adjacency matrix; otherwise, it should be a
        list of weights corresponding to the edges in ``edge_index``.
    normalise : bool (default True)
        Indicates that the Laplacian should be normalised using the degree
        matrix.
    topk : bool (default True)
        Indicates that the input is a top-k sparse matrix. Has no effect if
        ``W`` is not a sparse matrix.
    """
    W = delete_diagonal(W)
    deg = degree(W)
    D = vmap_over_outer(jnp.diagflat, 1)((deg,))
    L = D - W
    if normalise:
        norm_fac = jnp.where(deg == 0, 1, 1 / jnp.sqrt(deg))
        L = L * norm_fac[..., None, :]
        L = L * norm_fac[..., None]
        L = vmap_over_outer(partial(fill_diagonal, fill=1), 2)((L,))
    return L


@graph_laplacian.register
def _(
    W: TopKTensor,
    normalise: bool = True,
    topk: bool = True,
):
    if not topk:
        return _sparse_laplacian(W=W, normalise=normalise)
    W = topk_diagzero(W)
    deg = W.data.sum(-1, keepdims=True)
    L = topk_diagaugment(-W, deg)
    if normalise:
        norm_fac = jnp.where(deg == 0, 1, 1 / jnp.sqrt(deg)).squeeze(-1)
        L = dspdmm(L, norm_fac)
    return L


def _sparse_delete_selfloops(
    edge_index: Tensor,
    edge_weight: Tensor
) -> Tensor:
    mask = edge_index[..., 0] == edge_index[..., 1]
    return edge_index, edge_weight.at[mask].set(0)


def _sparse_append_selfloops(
    edge_index: Tensor,
    edge_weight: Tensor,
    fill_value: Union[float, Tensor],
    num_nodes: int
) -> Tensor:
    loop_index = jnp.arange(0, num_nodes, dtype=jnp.int32)
    loop_index = jnp.tile(loop_index[..., None], (1, 2))
    edge_index = jnp.concatenate([edge_index, loop_index], axis=-2)
    if isinstance(fill_value, float) or isinstance(fill_value, int):
        edge_weight = jnp.concatenate([
            edge_weight, fill_value * jnp.ones((num_nodes,))
        ], axis=-1)
    else:
        edge_weight = jnp.concatenate([edge_weight, fill_value], axis=-1)
    return edge_index, edge_weight


def _sparse_laplacian(W: Any, normalise: bool = True) -> Any:
    num_nodes = W.shape[-1]
    edge_index, edge_weight = W.indices, W.data
    edge_index, edge_weight = _sparse_delete_selfloops(edge_index, edge_weight)
    row, col = edge_index[..., 0], edge_index[..., 1]
    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,)
    )
    deg = jax.lax.scatter_add(
        jnp.zeros(num_nodes),
        row[..., None],
        edge_weight,
        dnums
    )
    if normalise:
        deg_inv_sqrt = jnp.power(deg, -0.5)
        deg_inv_sqrt = jnp.where(
            jnp.logical_or(jnp.isnan(deg_inv_sqrt), jnp.isinf(deg_inv_sqrt)),
            0,
            deg_inv_sqrt)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        edge_index, edge_weight = _sparse_append_selfloops(
            edge_index,
            edge_weight=-edge_weight,
            fill_value=1,
            num_nodes=num_nodes
        )
    else:
        edge_index, edge_weight = _sparse_append_selfloops(
            edge_index,
            edge_weight=-edge_weight,
            fill_value=deg,
            num_nodes=num_nodes
        )
    return BCOO((edge_weight, edge_index), shape=W.shape)
