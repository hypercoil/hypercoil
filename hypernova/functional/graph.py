# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Graph measures
~~~~~~~~~~~~~~
Measures on graphs and networks.
"""
import torch


def girvan_newman_null(A):
    """
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

    Dimension
    ---------
    - Input: :math:`(N, *, I, O)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      I denotes number of vertices in the source set and O denotes number of
      vertices in the sink set. If the same set of vertices emits and receives
      edges, then :math:`I = O`.
    - Output: :math:`(N, *, I, O)`

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
    k_i = A.sum(-1, keepdim=True)
    k_o = A.sum(-2, keepdim=True)
    two_m = k_i.sum(-2, keepdim=True)
    return k_i @ k_o / two_m


def modularity_matrix(A, gamma=1, null=girvan_newman_null,
                      normalise=False, **params):
    """
    Modularity matrices for a tensor block.

    The modularity matrix is defined as a normalised, weighted difference
    between the adjacency matrix and a suitable null model. For a weight
    :math:`\gamma`, an adjacency matrix :math:`A`, a null model :math:`P`, and
    total edge weight :math:`2m`, the modularity matrix is computed as

    :math:`B = \frac{1}{2m} \left( A - \gamma P \right)`

    Dimension
    ---------
    - Input: :math:`(N, *, I, O)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      I denotes number of vertices in the source set and O denotes number of
      vertices in the sink set. If the same set of vertices emits and receives
      edges, then :math:`I = O`.
    - Output: :math:`(N, *, I, O)`

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
    null : callable(A) (default `girvan_newman_null`)
        Function of A that returns, for each adjacency matrix in the input
        tensor block, a suitable null model.
    normalise : bool (default False)
        Indicates that the resulting matrix should be normalised by the total
        matrix degree. This may not be necessary for many use cases -- for
        instance, where the arg max of a function of the modularity matrix is
        desired.
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
    mod = A - gamma * null(A, **params)
    if normalise:
        two_m = A.sum([-2, -1], keepdim=True)
        return mod / two_m
    return mod


def coaffiliation(C_i, C_o=None, L=None, exclude_diag=True):
    C_o = C_o or C_i
    if L is None:
        C = C_i @ C_o.transpose(-1, -2)
    else:
        C = C_i @ L @ C_o.transpose(-1, -2)
    if exclude_diag:
        C[torch.eye(C.size(-1), dtype=torch.bool)] = 0
    return C


def relaxed_modularity(A, C, C_o=None, L=None, exclude_diag=True, gamma=1,
                       null=girvan_newman_null, normalise=True, **params):
    B = modularity_matrix(A, gamma=gamma, null=null,
                          normalise=normalise, **params)
    C = coaffiliation(C, C_o=C_o, L=L, exclude_diag=exclude_diag)
    return (B * C).sum([-2, -1])
