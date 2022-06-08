# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Maximum potential bipartite lattice (MPBL) algorithm.
"""
import torch
import numpy as np
from torch.nn.functional import softmax
from .base import DomainInitialiser
from ..functional import pairedcorr, delete_diagonal, sym2vec, vec2sym


def corr_criterion(orig, recon, u):
    """
    Select the compression scheme that maximises the feature-wise
    correlation between the original uncompressed matrix and its
    reconstruction.
    """
    q = -pairedcorr(sym2vec(recon).view(1, -1), sym2vec(orig).view(1, -1))
    if torch.isnan(q): q = 0
    return q


def potential_criterion(orig, recon, u):
    """
    Select the compression scheme that maximises the transmitted
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
    candidates = torch.ones(n_in, dtype=torch.bool)
    candidates_ids = torch.arange(n_in)
    asgt_u = torch.zeros(n_out)
    asgt = torch.zeros((n_out, n_in), dtype=torch.bool)

    potentials = delete_diagonal(potentials_orig)
    for i in range(n_out):
        if torch.sum(candidates) == 0:
            candidates = torch.ones(n_in, dtype=torch.bool)

        n_edges = 0
        select = _init_select(candidates, candidates_ids,
                              potentials, random_init)
        n_edges = _update_assignment(asgt, candidates, select, n_edges, i)

        while n_edges < n_edges_out:
            if torch.sum(candidates) == 0:
                candidates = torch.ones(n_in, dtype=torch.bool)
            select = _select_edge(candidates, candidates_ids,
                                  potentials, asgt[i, :], temperature)
            n_edges = _update_assignment(asgt, candidates, select, n_edges, i)
        idx = torch.nonzero(asgt[i, :]).squeeze()
        asgt_u[i] = torch.sum(potentials[idx.view(-1, 1), idx])
        potentials[idx.view(-1, 1), idx] /= attenuation
    return asgt.to(dtype=torch.float), asgt_u


def _mpbl_eval(criterion, asgt, objective, n_edges_in, n_edges_out,
               asgt_u, crit_u=float('inf'), max_asgt=None):
    """
    Evaluate a single run of the MPBL algorithm.
    """
    # TODO: this normalisation is incorrect. It approximately works for the
    # correlation criterion but will break for others.
    compressed = (asgt[0] / n_edges_out[0]) @ (objective @
        (asgt[1] / n_edges_out[1]).t())
    recon = (asgt[0] / n_edges_in[0]).t() @ (
        compressed @ (asgt[1] / n_edges_in[1]))
    u = criterion(objective, recon, asgt_u)
    if u < crit_u:
        crit_u = u
        max_asgt = asgt
    return u, crit_u, max_asgt


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
        n_in = (potentials[0].size(0), potentials[1].size(0))
        potentials_orig = (potentials[0].clone(), potentials[1].clone())
        n_edges_allowed = [order * np.lcm(i, o)
                           for i, o in zip(n_in, n_out)]
        n_edges_out = [a // o for a, o in zip(n_edges_allowed, n_out)]
        n_edges_in = [a // i for a, i in zip(n_edges_allowed, n_in)]
        max_asgt = [torch.ones((n_out[0], n_in[0])),
                    torch.ones((n_out[1], n_in[1]))]
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
            asgt, asgt_u = _mpbl_run(n_in, n_out, potentials_orig,
                                     random_init, n_edges_out,
                                     temperature, attenuation)
            u, crit_u, max_asgt = _mpbl_eval(
                asgt=(asgt, asgt),
                criterion=criterion, objective=objective,
                n_edges_in=(n_edges_in, n_edges_in),
                n_edges_out=(n_edges_out, n_edges_out),
                asgt_u=asgt_u, crit_u=crit_u, max_asgt=max_asgt
            )
        else:
            asgt, asgt_u = _mpbl_run(n_in[0], n_out[0], potentials_orig[0],
                                     random_init, n_edges_out[0],
                                     temperature, attenuation)
            u, crit_u[0], max_asgt = _mpbl_eval(
                asgt=(asgt, max_asgt[1]),
                criterion=criterion, objective=objective,
                n_edges_in=n_edges_in, n_edges_out=n_edges_out,
                asgt_u=asgt_u, crit_u=crit_u[0], max_asgt=max_asgt
            )
            asgt, asgt_u = _mpbl_run(n_in[1], n_out[1], potentials_orig[1],
                                     random_init, n_edges_out[1],
                                     temperature, attenuation)
            u, crit_u[1], max_asgt = _mpbl_eval(
                asgt=(max_asgt[0], asgt),
                criterion=criterion, objective=objective,
                n_edges_in=n_edges_in, n_edges_out=n_edges_out,
                asgt_u=asgt_u, crit_u=crit_u[1], max_asgt=max_asgt
            )
    if symmetric:
        max_asgt = max_asgt[0]
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


#TODO: a lot of this is not correctly implemented. Revisit when we work on
# sylo net.
class BipartiteLatticeInit(DomainInitialiser):
    r"""
    Initialiser for sparse and biregular compression matrices.

    .. warning::
        Much of this functionality is currently wrong. But then, there isn't
        really any reason to use this initialiser right now anyway.

    Parameters
    ----------
    n_out: int or (int, int)
        Number of output vertices. Ideally, this should be selected such that
        there are many common multiples of ``n_out`` and the number of
        vertices in the input potentials.
    order: int
        Lattice order of the bipartition.

        - 1: total number of edges is the least common multiple of ``n_in``
          and ``n_out``
        - 2: total number of edges is twice the least common multiple
    potentials: Tensor or (Tensor, Tensor)
        Array of potentials to be maximised in the bipartition. This could,
        for instance, be the mutual information among vertices or the
        community allegiance matrix.

        Size: :math:`n_{in} \times n_{in}`

        If this is a tuple, then the algorithm will solve two different MPBL
        problems: one for each tensor. In this case, the first tensor should
        be of dimension :math:`H_{in} \times H_{in}` and the second should be
        of dimension :math:`W_{in} \times W_{in}`, where the dimension of the
        ``objective`` matrix is :math:`H_{in} \times W_{in}`.
    objective: Tensor
        Tensor whose reconstruction from vertical compression determines the
        best bipartite lattice. If none is provided, the algorithm defaults to
        the input potentials matrix.
    n_lattices : int
        Number of unique compression lattices to initialise.
    residualise : bool
        If this is true and multiple lattices are specified, the objective for
        each lattice after the first is residualised with respect to the
        reconstructed (compressed and then uncompressed) objective tensor.
    svd : bool
        If this is true, then the potentials for each succeeding lattice are
        set using the next principal component of the potentials.
    channel_multiplier : int
        Number of outputs for each lattice.
    sign : ``'+'`` or ``'-'``
        Initialises the ``sign`` parameter of a vertical compression module.
    iters : int
        Number of iterations of the greedy approach to run. Only the result
        that maximises the final potential will be returned.
    temperature : float
        Softmax temperature for selecting the next vertex to fuse.
        0 deterministically fuses the maximum (default).
        Infinity fuses randomly.
    random_init : bool
        Procedural initialisation. True begins from a random vertex (default).
        False begins from a vertex connected to the edge with the maximum
        overall potential.
    attenuation : int
        Attenuation factor for the potentials matrix. The potentials joining
        each vertex set that is compressed into a single new vertex are
        divided by the attenuation factor. This helps to prevent redundancy in
        compressed sets.
    next : None or ``BipartiteLatticeInit`` instance
        If this is specified, the designated instance receives the compressed
        potentials as input.
    prev : None or ``BipartiteLatticeInit`` instance
        If this is specified, then the current instance is designated as
        ``next`` for the designated instance.
    domain : Domain object
        Used in conjunction with an activation function to constrain or
        transform the values of the initialised tensor.
    """
    def __init__(self, n_out, order,
                 potentials=None, objective=None,
                 n_lattices=1, residualise=False, svd=False,
                 channel_multiplier=1, sign='+',
                 iters=10, temperature=0,
                 random_init=True, attenuation=2,
                 next=None, prev=None, domain=None):
        self.n_out = n_out
        self.order = order
        self.iters = iters
        self.n_lattices = n_lattices
        self.residualise = residualise
        self.svd = svd
        self.channel_multiplier = channel_multiplier
        self.sign = sign
        self.temperature = temperature
        self.random_init = random_init
        self.attenuation = attenuation
        self.set_next(next)
        if prev is not None:
            prev.set_next(self)
        if potentials is not None:
            self.set_potentials(potentials)
        if objective is not None:
            self.set_objective(objective)
        else:
            self.objective = None
        super(BipartiteLatticeInit, self).__init__(init=None, domain=domain)

    def set_next(self, other):
        self.next = other

    def set_potentials(self, potentials, sign='+'):
        if self.svd:
            potential_vec = sym2vec(potentials)
            _, _, v = torch.svd(potential_vec)
            v = v.t()[:self.n_lattices]
            for i, vec in enumerate(v):
                if (self.sign == '+' and
                    pairedcorr(vec, potential_vec).sum() < 0):
                    v[i] = -vec
                if (self.sign == '-' and
                    pairedcorr(vec, potential_vec).sum() > 0):
                    v[i] = -vec
            potentials = vec2sym(v)
        if isinstance(potentials, torch.Tensor) and (potentials.dim() == 2):
            potentials = [potentials] * self.n_lattices
        if sign == self.sign:
            self.potentials = potentials
        else:
            self.potentials = [-p for p in potentials]

    def set_objective(self, objective, sign='+'):
        if sign == self.sign:
            self.objective = objective
        else:
            self.objective = -objective

    def sign_vector(self):
        base = torch.ones((self.n_lattices * self.channel_multiplier, 1, 1))
        if self.sign == '+':
            return base
        elif self.sign == '-':
            return -base

    def __call__(self, tensor, mask):
        assert tensor.shape == mask.shape, (
            'Tensor and mask shapes must match')
        if tensor.dim() > 2:
            assert tensor.shape[-3] == (
                self.n_lattices * self.channel_multiplier), (
                'Tensor dimension does not match lattice count')
        if self.residualise and self.svd:
            raise ValueError('Cannot set both `residualise` and `svd`')
        U_prop = []
        recon = torch.empty((0, tensor.shape[-1], tensor.shape[-1]))
        max_asgt = None
        for i in range(self.n_lattices):
            if self.residualise and (i != 0):
                compressed = max_asgt @ self.potentials @ max_asgt.T
                recon = torch.cat(
                    (recon,
                    (max_asgt.T @ compressed @ max_asgt).mean(
                        0, keepdim=True)),
                    dim=0)
                scale = torch.linalg.lstsq(
                    sym2vec(recon).view(i, -1).transpose(-2, -1),
                    sym2vec(self.potentials[0]).view(-1, 1)
                ).solution
                potentials = self.potentials[0] - (
                    recon.transpose(0, -1) @ scale).squeeze()
            else:
                potentials = self.potentials[i]
            max_asgt, _, u_prop = maximum_potential_bipartite_lattice(
                potentials=potentials,
                n_out=self.n_out,
                order=self.order,
                iters=self.iters,
                temperature=self.temperature,
                random_init=self.random_init,
                attenuation=self.attenuation,
                objective=self.objective,
                criterion=corr_criterion
            )
            U_prop += [u_prop]
            with torch.no_grad():
                if tensor.dim() > 2:
                    begin = i * self.channel_multiplier
                    end = begin + self.channel_multiplier
                    tensor[begin:end] = self.domain.preimage(max_asgt)
                    mask[begin:end] = max_asgt
                else:
                    tensor[:] = self.domain.preimage(max_asgt)
                    mask[:] = max_asgt
        if self.next is not None:
            self.next.set_potentials(U_prop, sign=self.sign)
