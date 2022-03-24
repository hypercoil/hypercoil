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
    k_i = A.sum(-1, keepdim=True)
    k_o = A.sum(-2, keepdim=True)
    two_m = k_i.sum(-2, keepdim=True)
    return k_i @ k_o / two_m


def modularity_matrix(A, gamma=1, null=girvan_newman_null,
                      normalise=False, sign='+', **params):
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
    null : callable(A) (default `girvan_newman_null`)
        Function of A that returns, for each adjacency matrix in the input
        tensor block, a suitable null model.
    normalise : bool (default False)
        Indicates that the resulting matrix should be normalised by the total
        matrix degree. This may not be necessary for many use cases -- for
        instance, where the arg max of a function of the modularity matrix is
        desired.
    sign : '+', '-', or None (default '+')
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
        A = torch.relu(A)
    elif sign == '-':
        A = -torch.relu(-A)
    mod = A - gamma * null(A, **params)
    if normalise:
        two_m = A.sum([-2, -1], keepdim=True)
        return mod / two_m
    return mod


def coaffiliation(C_i, C_o=None, L=None, exclude_diag=True, normalise=False):
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
        assumed that the source and sink sets are the same, and `C_o` is set
        equal to `C_i`.
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
        norm_fac_i = torch.max(torch.tensor(1), C_i.max())
        norm_fac_o = torch.max(torch.tensor(1), C_o.max())
        C_i = C_i / norm_fac_i
        C_o = C_o / norm_fac_o
    if L is None:
        C = C_i @ C_o.transpose(-1, -2)
    else:
        C = C_i @ L @ C_o.transpose(-1, -2)
    if exclude_diag:
        C[torch.eye(C.size(-1), dtype=torch.bool, device=C.device)] = 0
    return C


def relaxed_modularity(A, C, C_o=None, L=None, exclude_diag=True, gamma=1,
                       null=girvan_newman_null, normalise_modularity=True,
                       normalise_coaffiliation=True, directed=False, sign='+',
                       **params):
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
        assumed that the source and sink sets are the same, and `C_o` is set
        equal to `C`.
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
        Resolution parameter for the modularity matrix. A smaller value assigns
        maximum modularity to partitions with large communities, while a larger
        value assigns maximum modularity to partitions with many small
        communities.
    null : callable(A) (default `girvan_newman_null`)
        Function of A that returns, for each adjacency matrix in the input
        tensor block, a suitable null model.
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
    sign : '+', '-', or None (default '+')
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
    Q = (B * C).sum([-2, -1])
    if not directed:
        return Q / 2
    return Q
