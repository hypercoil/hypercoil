# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Maximum potential bipartite lattice (MPBL) algorithm.
"""
import jax
import jax.numpy as jnp
import distrax
from typing import Any, Callable, Optional, Tuple, Union
from jax.nn import softmax
from .base import MappedInitialiser
from ..engine import Tensor
from ..functional import (
    delete_diagonal,
    pairedcorr,
    sym2vec,
    vec2sym
)


def corr_criterion(
    orig: Tensor,
    recon: Tensor,
    u: Any,
) -> float:
    """
    Select the compression scheme that maximises the feature-wise
    correlation between the original uncompressed matrix and its
    reconstruction.
    """
    q = -pairedcorr(
        sym2vec(recon).reshape(1, -1),
        sym2vec(orig).reshape(1, -1)
    )
    if jnp.isnan(q): q = 0
    return q


def potential_criterion(
    orig: Any,
    recon: Any,
    u: Tensor,
) -> float:
    """
    Select the compression scheme that maximises the transmitted
    potential.
    """
    return -jnp.sum(u)


def propagate_potentials(
    potentials: Tensor,
    mask: Tensor,
) -> Tensor:
    """Propagate potentials to the compressed space."""
    prop = mask / mask.sum(1)[0]
    return (prop @ potentials) @ prop.T


#TODO: what the hell is this denominator?
def propagate_matrix(
    A: Tensor,
    mask_L: Tensor,
    mask_R: Tensor,
) -> Tensor:
    prop_L = mask_L / mask_L.sum(1)[0]
    prop_R = mask_R / mask_R.sum(1)[0]
    return (prop_L @ A) @ prop_R.T


def _init_select(
    candidates: Tensor,
    candidates_ids: Tensor,
    potentials: Tensor,
    random_init: bool,
    key: 'jax.random.PRNGKey',
) -> int:
    """
    Make an initial selection of two input vertices to fuse into a new
    output vertex.
    """
    init_idx = jnp.nonzero(candidates)[0]
    if random_init:
        select = jax.random.randint(
            key=key,
            shape=(),
            minval=0,
            maxval=jnp.sum(candidates)
        )
    else:
        selection_pool = potentials[init_idx, init_idx]
        select = jnp.argmax(selection_pool)
        select = (select % selection_pool.shape[0]).astype(int)
    return candidates_ids[candidates][select]


def _select_edge(
    candidates: Tensor,
    candidates_ids: Tensor,
    potentials: Tensor,
    asgt: Tensor,
    temperature: float,
    key: 'jax.random.PRNGKey',
) -> int:
    """
    Select the next input vertex to fuse into the current output vertex.
    """
    u_sum = jnp.sum(
        potentials[asgt, :][:, candidates].reshape(
        jnp.sum(asgt), -1), 0)
    probs = softmax(u_sum / temperature, axis=0)
    select = distrax.Categorical(probs=probs).sample(seed=key, sample_shape=())
    return candidates_ids[candidates][select]


def _update_assignment(
    asgt: Tensor,
    candidates: Tensor,
    select: int,
    n_edges: int,
    out_idx: int,
) -> int:
    """
    Update the assignment and the index of available candidates.
    """
    candidates = candidates.at[select].set(0)
    asgt = asgt.at[out_idx, select].set(1)
    return n_edges + 1, candidates, asgt


def _mpbl_run(
    n_in: int,
    n_out: int,
    potentials_orig: Tensor,
    random_init: bool,
    n_edges_out: int,
    temperature: float,
    attenuation: float,
    key: 'jax.random.PRNGKey',
):
    """
    Execute a single run of the MPBL algorithm.
    """
    candidates = jnp.ones(n_in, dtype=bool)
    candidates_ids = jnp.arange(n_in, dtype=int)
    asgt_u = jnp.zeros(n_out)
    asgt = jnp.zeros((n_out, n_in), dtype=bool)

    potentials = delete_diagonal(potentials_orig)
    for i in range(n_out):
        key, init_key, select_key = jax.random.split(key, 3)
        if jnp.sum(candidates) == 0:
            candidates = jnp.ones(n_in, dtype=bool)

        n_edges = 0
        select = _init_select(candidates, candidates_ids,
                              potentials, random_init, key=init_key)
        n_edges, candidates, asgt = _update_assignment(
            asgt, candidates, select, n_edges, i
        )

        while n_edges < n_edges_out:
            select_key = jax.random.split(select_key, 1)[0]
            if jnp.sum(candidates) == 0:
                candidates = jnp.ones(n_in, dtype=bool)
            select = _select_edge(candidates, candidates_ids,
                                  potentials, asgt[i, :], temperature,
                                  key=select_key)
            n_edges, candidates, asgt = _update_assignment(
                asgt, candidates, select, n_edges, i
            )
        idx = jnp.nonzero(asgt[i, :])[0]
        print(idx.shape, asgt_u.shape, potentials.shape)
        asgt_u = asgt_u.at[i].set(jnp.sum(potentials_orig[idx, idx]))
        potentials = potentials.at[idx, idx].set(
            potentials[idx, idx] / attenuation)
        #asgt_u[i] = jnp.sum(potentials[idx.reshape(-1, 1), idx])
        #potentials[idx.reshape(-1, 1), idx] /= attenuation
    return asgt.astype(float), asgt_u


def _mpbl_eval(
    criterion: Callable,
    asgt: Tensor,
    objective: Tensor,
    n_edges_in: int,
    n_edges_out: int,
    asgt_u: Tensor,
    crit_u: float = float('inf'),
    max_asgt: Tensor = None
):
    """
    Evaluate a single run of the MPBL algorithm.
    """
    # TODO: this normalisation is incorrect. It approximately works for the
    # correlation criterion but will break for others.
    print(asgt[0].shape, asgt[1].shape, objective.shape)
    compressed = (asgt[0] / n_edges_out[0]) @ (objective @
        (asgt[1] / n_edges_out[1]).swapaxes(-2, -1))
    print(compressed.shape)
    recon = (asgt[0] / n_edges_in[0]).swapaxes(-2, -1) @ (
        compressed @ (asgt[1] / n_edges_in[1]))
    u = criterion(objective, recon, asgt_u)
    if u < crit_u:
        crit_u = u
        max_asgt = asgt
    return u, crit_u, max_asgt


def maximum_potential_bipartite_lattice(
    potentials: Union[Tensor, Tuple[Tensor, Tensor]],
    n_out: Union[int, Tuple[int, int]],
    order: int,
    iters: int = 100,
    temperature: float = 0,
    random_init: bool = True,
    attenuation: float = 2.,
    objective: Optional[Tensor] = None,
    criterion: Callable = corr_criterion,
    *,
    key: 'jax.random.PRNGKey',
) -> Tuple[Union[Tensor, Tuple[Tensor, Tensor]], Tensor, Tensor]:
    r"""
    Estimates the maximum potential bipartite lattice using a greedy Monte
    Carlo approach. A naive solution to the vertical compression problem.

    .. warning::

        This function is experimental and currently incompatible with
        JAX JIT compilation.

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
    attenuation: float
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
        n_in = (potentials[0].shape[0], potentials[1].shape[0])
        potentials_orig = (potentials[0].copy(), potentials[1].copy())
        n_edges_allowed = [order * jnp.lcm(i, o)
                           for i, o in zip(n_in, n_out)]
        n_edges_out = [a // o for a, o in zip(n_edges_allowed, n_out)]
        n_edges_in = [a // i for a, i in zip(n_edges_allowed, n_in)]
        max_asgt = [jnp.ones((n_out[0], n_in[0])),
                    jnp.ones((n_out[1], n_in[1]))]
        U_prop = (None, None)
        crit_u = [float('inf'), float('inf')]
    else:
        symmetric = True
        n_in, _ = potentials.shape
        potentials_orig = potentials.clone()
        n_edges_allowed = order * jnp.lcm(n_in, n_out)
        n_edges_out = n_edges_allowed // n_out
        n_edges_in = n_edges_allowed // n_in
        max_asgt = None
        U_prop = None
        crit_u = float('inf')
    if temperature == 0:
        temperature = jnp.finfo(potentials[0].dtype).tiny
    if objective is None:
        objective = potentials_orig
    for _ in range(iters):
        key = jax.random.split(key, 1)[0]
        if symmetric:
            asgt, asgt_u = _mpbl_run(n_in, n_out, potentials_orig,
                                     random_init, n_edges_out,
                                     temperature, attenuation,
                                     key=key)
            u, crit_u, max_asgt = _mpbl_eval(
                asgt=(asgt, asgt),
                criterion=criterion, objective=objective,
                n_edges_in=(n_edges_in, n_edges_in),
                n_edges_out=(n_edges_out, n_edges_out),
                asgt_u=asgt_u, crit_u=crit_u, max_asgt=max_asgt
            )
        else:
            key_L, key_R = jax.random.split(key)
            asgt, asgt_u = _mpbl_run(n_in[0], n_out[0], potentials_orig[0],
                                     random_init, n_edges_out[0],
                                     temperature, attenuation,
                                     key=key_L)
            u, crit_u[0], max_asgt = _mpbl_eval(
                asgt=(asgt, max_asgt[1]),
                criterion=criterion, objective=objective,
                n_edges_in=n_edges_in, n_edges_out=n_edges_out,
                asgt_u=asgt_u, crit_u=crit_u[0], max_asgt=max_asgt
            )
            asgt, asgt_u = _mpbl_run(n_in[1], n_out[1], potentials_orig[1],
                                     random_init, n_edges_out[1],
                                     temperature, attenuation,
                                     key=key_R)
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
    potentials: Union[Tensor, Tuple[Tensor, Tensor]],
    n_out: Union[int, Tuple[int, int]],
    order_min: int = 1,
    order_max: int = 10,
    iters: int = 100,
    temperature: float = 0.,
    random_init: bool = True,
    objective: Optional[Tensor] = None,
    attenuation: float = 2,
    inner_criterion: Callable = corr_criterion,
    outer_criterion: Callable = corr_criterion
) -> Tuple[Any, ...]:
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
            compressed = (out[0] @ objective) @ out[0].swapaxes(-2, -1)
            recon = out[0].astype(float).swapaxes(-2, -1) @ (
                compressed @ out[0].astype(float))
            compression_error[i] = outer_criterion(objective, recon, None)
        else:
            compressed = (out[0][0] @ objective) @ out[0][1].swapaxes(-2, -1)
            recon = out[0][0].swapaxes(-2, -1) @ (compressed @ out[0][1])
            compression_error[i] = outer_criterion(objective, recon, None)

    idx = jnp.argmin(compression_error)
    order = orders[idx]
    return omax_asgt[idx], omax_u[idx], oU_prop[idx], order, compression_error


#TODO: a lot of this is not correctly implemented. Revisit when we work on
# sylo net.
class BipartiteLatticeInit(MappedInitialiser):
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
        raise NotImplementedError(
            'This initialiser is not yet implemented for public use.')
    #     self.n_out = n_out
    #     self.order = order
    #     self.iters = iters
    #     self.n_lattices = n_lattices
    #     self.residualise = residualise
    #     self.svd = svd
    #     self.channel_multiplier = channel_multiplier
    #     self.sign = sign
    #     self.temperature = temperature
    #     self.random_init = random_init
    #     self.attenuation = attenuation
    #     self.set_next(next)
    #     if prev is not None:
    #         prev.set_next(self)
    #     if potentials is not None:
    #         self.set_potentials(potentials)
    #     if objective is not None:
    #         self.set_objective(objective)
    #     else:
    #         self.objective = None
    #     super(BipartiteLatticeInit, self).__init__(init=None, domain=domain)

    # def set_next(self, other):
    #     self.next = other

    # def set_potentials(self, potentials, sign='+'):
    #     if self.svd:
    #         potential_vec = sym2vec(potentials)
    #         _, _, v = torch.svd(potential_vec)
    #         v = v.t()[:self.n_lattices]
    #         for i, vec in enumerate(v):
    #             if (self.sign == '+' and
    #                 pairedcorr(vec, potential_vec).sum() < 0):
    #                 v[i] = -vec
    #             if (self.sign == '-' and
    #                 pairedcorr(vec, potential_vec).sum() > 0):
    #                 v[i] = -vec
    #         potentials = vec2sym(v)
    #     if isinstance(potentials, torch.Tensor) and (potentials.dim() == 2):
    #         potentials = [potentials] * self.n_lattices
    #     if sign == self.sign:
    #         self.potentials = potentials
    #     else:
    #         self.potentials = [-p for p in potentials]

    # def set_objective(self, objective, sign='+'):
    #     if sign == self.sign:
    #         self.objective = objective
    #     else:
    #         self.objective = -objective

    # def sign_vector(self):
    #     base = torch.ones((self.n_lattices * self.channel_multiplier, 1, 1))
    #     if self.sign == '+':
    #         return base
    #     elif self.sign == '-':
    #         return -base

    # def __call__(self, tensor, mask):
    #     assert tensor.shape == mask.shape, (
    #         'Tensor and mask shapes must match')
    #     if tensor.dim() > 2:
    #         assert tensor.shape[-3] == (
    #             self.n_lattices * self.channel_multiplier), (
    #             'Tensor dimension does not match lattice count')
    #     if self.residualise and self.svd:
    #         raise ValueError('Cannot set both `residualise` and `svd`')
    #     U_prop = []
    #     recon = torch.empty((0, tensor.shape[-1], tensor.shape[-1]))
    #     max_asgt = None
    #     for i in range(self.n_lattices):
    #         if self.residualise and (i != 0):
    #             compressed = max_asgt @ self.potentials @ max_asgt.T
    #             recon = torch.cat(
    #                 (recon,
    #                 (max_asgt.T @ compressed @ max_asgt).mean(
    #                     0, keepdim=True)),
    #                 dim=0)
    #             scale = torch.linalg.lstsq(
    #                 sym2vec(recon).view(i, -1).transpose(-2, -1),
    #                 sym2vec(self.potentials[0]).view(-1, 1)
    #             ).solution
    #             potentials = self.potentials[0] - (
    #                 recon.transpose(0, -1) @ scale).squeeze()
    #         else:
    #             potentials = self.potentials[i]
    #         max_asgt, _, u_prop = maximum_potential_bipartite_lattice(
    #             potentials=potentials,
    #             n_out=self.n_out,
    #             order=self.order,
    #             iters=self.iters,
    #             temperature=self.temperature,
    #             random_init=self.random_init,
    #             attenuation=self.attenuation,
    #             objective=self.objective,
    #             criterion=corr_criterion
    #         )
    #         U_prop += [u_prop]
    #         with torch.no_grad():
    #             if tensor.dim() > 2:
    #                 begin = i * self.channel_multiplier
    #                 end = begin + self.channel_multiplier
    #                 tensor[begin:end] = self.domain.preimage(max_asgt)
    #                 mask[begin:end] = max_asgt
    #             else:
    #                 tensor[:] = self.domain.preimage(max_asgt)
    #                 mask[:] = max_asgt
    #     if self.next is not None:
    #         self.next.set_potentials(U_prop, sign=self.sign)
