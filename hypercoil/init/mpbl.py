# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Maximum potential bipartite lattice (MPBL) algorithm.
"""
import torch
import numpy as np
from ..functional import pairedcorr
from torch.nn.functional import softmax


def corr_criterion(orig, recon, u):
    """
    Selects the compression scheme that maximises the feature-wise
    correlation between the original uncompressed matrix and its
    reconstruction.
    """
    q = -pairedcorr(sym2vec(recon).view(1, -1), sym2vec(orig).view(1, -1))
    if torch.isnan(q): q = 0
    return q


def potential_criterion(orig, recon, u):
    """
    Selects the compression scheme that maximises the transmitted
    potential.
    """
    return -torch.sum(u)


def propagate_potentials(potentials, mask):
    """Propagate potentials to the compressed space."""
    prop = mask / mask.sum(1)[0]
    return (prop @ potentials) @ prop.t()


def propagate_matrix(A, mask_L, mask_R):
    prop_L = mask_L / mask_L.sum(1)[0]
    prop_R = mask_R / mask_R.sum(1)[0]
    return (prop_L @ A) @ prop_R.t()


def _init_select(candidates, candidates_ids, potentials, random_init):
    """
    Make an initial selection of two input vertices to fuse into a new
    output vertex.
    """
    init_idx = torch.nonzero(candidates).squeeze()
    if random_init:
        select = int(torch.sum(candidates) * torch.rand(1))
    else:
        selection_pool = potentials[init_idx.view(-1, 1), init_idx]
        select = torch.argmax(selection_pool)
        select = (select % selection_pool.size(0)).long()
    return candidates_ids[candidates][select]


def _select_edge(candidates, candidates_ids, potentials, asgt, temperature):
    """
    Select the next input vertex to fuse into the current output vertex.
    """
    u_sum = torch.sum(
        potentials[asgt, :][:, candidates].view(
        torch.sum(asgt), -1), 0)
    probs = softmax(input=u_sum / temperature, dim=0)
    select = torch.distributions.Categorical(probs).sample()
    return candidates_ids[candidates][select]


def _update_assignment(asgt, candidates, select, n_edges, out_idx):
    """
    Update the assignment and the index of available candidates.
    """
    candidates[select] = 0
    asgt[out_idx, select] = 1
    return n_edges + 1


def _mpbl_run(n_in, n_out, potentials_orig, random_init,
              n_edges_out, temperature, attenuation):
    """
    Execute a single run of the MPBL algorithm.
    """
    candidates = torch.ones(n_in).byte()
    candidates_ids = torch.arange(n_in)
    asgt_u = torch.zeros(n_out)
    asgt = torch.zeros(n_out, n_in).byte()

    potentials = potentials_orig - torch.diag(torch.diag(potentials_orig))
    for i in range(n_out):
        if torch.sum(candidates) == 0:
            candidates = torch.ones(n_in).byte()

        n_edges = 0
        select = _init_select(candidates, candidates_ids,
                              potentials, random_init)
        n_edges = _update_assignment(asgt, candidates, select, n_edges, i)

        while n_edges < n_edges_out:
            if torch.sum(candidates) == 0:
                candidates = torch.ones(n_in).byte()
            select = _select_edge(candidates, candidates_ids,
                                  potentials, asgt[i, :], temperature)
            n_edges = _update_assignment(asgt, candidates, select, n_edges, i)
        idx = torch.nonzero(asgt[i, :]).squeeze()
        asgt_u[i] = torch.sum(potentials[idx.view(-1, 1), idx])
        potentials[idx.view(-1, 1), idx] /= attenuation
    return asgt.float(), asgt_u


def maximum_potential_bipartite_lattice(potentials, n_out, order, iters=100,
                                        temperature=0, random_init=True,
                                        attenuation=2, objective=None,
                                        criterion=corr_criterion):
    r"""
    Estimates the maximum potential bipartite lattice using a greedy Monte
    Carlo approach. A naive solution to the vertical compression problem.

    Parameters
    ----------
    potentials: Tensor or (Tensor, Tensor)
        Array of potentials to be maximised in the bipartition. This could,
        for instance, be the mutual information among vertices or the
        community allegiance matrix.
        Size: `n_in` x `n_in`
        If this is a tuple, then the algorithm will solve two different MPBL
        problems: one for each tensor. In this case, the first tensor should
        be of dimension `H_in` x `H_in` and the second should be of dimension
        `W_in` x `W_in`, where the dimension of the `objective` matrix is
        `H_in` x `W_in`.
    n_out: int or (int, int)
        Number of output vertices. Ideally, this should be selected such that
        there are many common multiples of ``n_out`` and the number of
        vertices in the input potentials.
    order: int
        Lattice order of the bipartition.
        1: total number of edges is the least common multiple of `n_in` and
           `n_out`
        2: total number of edges is twice the least common multiple
    iters: int
        Number of iterations of the greedy approach to run. Only the result
        that maximises the final potential will be returned.
    temperature: float
        Softmax temperature for selecting the next vertex to fuse.
        0 deterministically fuses the maximum (default).
        Infinity fuses randomly.
    random_init: bool
        Procedural initialisation. True begins from a random vertex (default).
        False begins from a vertex connected to the edge with the maximum
        overall potential.
    attenuation: int
        Attenuation factor for the potentials matrix. The potentials joining
        each vertex set that is compressed into a single new vertex are
        divided by the attenuation factor. This helps to prevent redundancy in
        compressed sets.
    objective: Tensor
        Tensor whose reconstruction from vertical compression determines the
        best bipartite lattice. If none is provided, the algorithm defaults to
        the input potentials matrix.
    criterion: function
        Objective function that accepts three parameters and returns a scalar
        value.
        * `orig` is the original objective matrix.
        * `recon` is the reconstructed objective matrix
          (compressed and uncompressed).
        * `u` is the vector of potentials included in the compression.
        * The returned scalar is minimal for an optimum compression.

    Returns
    -------
    Tensor
        Assignment of nodes into a bipartition that maximises the potential of
        the resultant graph, among those tried by the greedy approach. This is
        not guaranteed to be a global optimum.
    float
        Overall potential of the bipartition.
    Tensor
        Propagated potentials. These can be maximised, for instance, in
        downstream vertical compressions.
    """
    if isinstance(potentials, tuple):
        symmetric = False
        n_in = (potentials[0].size(0), potentials[0].size(1))
        potentials_orig = (potentials[0].clone(), potentials[1].clone())
        n_edges_allowed = [order * np.lcm(i, o) for i, o in zip(n_in, n_out)]
        n_edges_out = [a // o for a, o in zip(n_edges_allowed, n_out)]
        n_edges_in = [a // i for a, i in zip(n_edges_allowed, n_in)]
        max_asgt = [torch.ones(n_out[0], n_in[0]), torch.ones(n_out[1], n_in[1])]
        U_prop = (None, None)
        crit_u = [float('inf'), float('inf')]
    else:
        symmetric = True
        n_in, _ = potentials.shape
        potentials_orig = potentials.clone()
        n_edges_allowed = order * np.lcm(n_in, n_out)
        n_edges_out = n_edges_allowed // n_out
        n_edges_in = n_edges_allowed // n_in
        max_asgt = None
        U_prop = None
        crit_u = float('inf')
    if temperature == 0:
        temperature = 1e-30
    if objective is None:
        objective = potentials_orig
    for _ in range(iters):
        if symmetric:
            asgt, asgt_u = _mpbl_run(n_in, n_out, potentials_orig, random_init,
                                     n_edges_out, temperature, attenuation)
            # TODO: this normalisation is incorrect
            compressed = (asgt / n_edges_out) @ (
                (asgt / n_edges_out) @ objective).t()
            recon = (asgt / n_edges_in).t() @ (compressed @ (asgt / n_edges_in))
            u = criterion(objective, recon, asgt_u)
            if u < crit_u:
                crit_u = u
                max_asgt = asgt
        else:
            asgt, asgt_u = _mpbl_run(n_in[0], n_out[0], potentials_orig[0],
                                     random_init, n_edges_out[0],
                                     temperature, attenuation)
            compressed = ((asgt / n_edges_out[0]) @ objective
                @ (max_asgt[1] / n_edges_out[1]).t())
            recon = (asgt / n_edges_in[0]).t() @ (
                compressed @ (max_asgt[1] / n_edges_in[1]))
            u = criterion(objective, recon, asgt_u)
            if u < crit_u[0]:
                crit_u[0] = u
                max_asgt[0] = asgt
            asgt, asgt_u = _mpbl_run(n_in[1], n_out[1], potentials_orig[1],
                                     random_init, n_edges_out[1],
                                     temperature, attenuation)
            compressed = ((max_asgt[0] / n_edges_out[0]) @ objective
                @ (asgt / n_edges_out[1]).t())
            recon = (max_asgt[0] / n_edges_in[0]).t() @ (
                compressed @ (asgt / n_edges_in[1]))
            u = criterion(objective, recon, asgt_u)
            if u < crit_u[1]:
                crit_u[1] = u
                max_asgt[1] = asgt
    if symmetric:
        U_prop = propagate_potentials(potentials_orig, max_asgt)
    else:
        U_prop = (propagate_potentials(potentials_orig[0], max_asgt[0]),
                  propagate_potentials(potentials_orig[1], max_asgt[1]))

    return max_asgt, crit_u, U_prop


def maximum_potential_bipartite_lattice_autoselect_order(
    potentials, n_out, order_min=1, order_max=10, iters=100,
    temperature=0, random_init=True, objective=None, attenuation=2,
    inner_criterion=corr_criterion, outer_criterion=corr_criterion):
    """
    Automatically selects the lattice order for a maximum potential
    bipartite lattice. That is to say, it runs the algorithm across a range
    of orders and selects the one that optimises a specified criterion.

    Parameters
    ----------
    order_min: int
        Minimum lattice order to consider.
    order_max: int
        Maximum lattice order to consider.
    inner_criterion: function
        Same as ``criterion`` in ``maximum_potential_bipartite_lattice``.
    outer_criterion: function
        Criterion to use for order selection. Only ``corr_criterion`` is
        currently supported.
    """
    symmetric = not isinstance(potentials, tuple)
    orders = range(order_min, order_max + 1)

    omax_asgt = [None for _ in orders]
    omax_u = [None for _ in orders]
    oU_prop = [None for _ in orders]
    compression_error = [None for _ in orders]

    if objective is None:
        objective = potentials

    for i, o in enumerate(orders):
        out = maximum_potential_bipartite_lattice(
            potentials=potentials, n_out=n_out, order=o, iters=iters,
            temperature=temperature, random_init=random_init,
            attenuation=attenuation, objective=objective,
            criterion=inner_criterion)
        omax_asgt[i] = out[0]
        omax_u[i] = out[1]
        oU_prop[i] = out[2]
        if symmetric:
            compressed = (out[0] @ objective) @ out[0].t()
            recon = out[0].float().t() @ (compressed @ out[0].float())
            compression_error[i] = outer_criterion(objective, recon, None)
        else:
            compressed = (out[0][0] @ objective) @ out[0][1].t()
            recon = out[0][0].t() @ (compressed @ out[0][1])
            compression_error[i] = outer_criterion(objective, recon, None)

    idx = np.argmin(compression_error)
    order = orders[idx]
    return omax_asgt[idx], omax_u[idx], oU_prop[idx], order, compression_error
