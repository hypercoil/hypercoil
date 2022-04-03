# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Modularity penalty
~~~~~~~~~~~~~~~~~~
Penalise a weight according to the quality of community structure it induces.
"""
import torch
from torch.nn import Module
from .base import Loss
from ..functional.graph import relaxed_modularity, girvan_newman_null


class ModularityLoss(Loss):
    r"""
    Differentiable relaxation of the Girvan-Newman modularity.

    This relaxation supports non-deterministic assignments of vertices to
    communities and non-assortative linkages between communities. It reverts
    to standard behaviour when the inputs it is provided are standard.

    The relaxed modularity is defined as the sum of all entries in the
    Hadamard (elementwise) product between the modularity matrix and the
    coaffiliation matrix.

    :math:`Q = \mathbf{1}^\intercal \left( B \circ H \right) \mathbf{1}`

    Penalising this favours a weight that induces a modular community
    structure on the input matrix -- or, an input matrix whose structure is
    reasonably accounted for by the proposed community affiliation weights.

    Parameters
    ----------
    nu : float (default 1)
        Loss function weight multiplier.
    affiliation_xfm : callable or None (default None)
        Transformation operating on the affiliation matrices (for instance, a
        softmax). This transformation is precomposed with the relaxed
        modularity.
    exclude_diag : bool (default True)
        Exclude weights along the diagonal (i.e., self-loops) when computing
        the modularity.
    gamma : nonnegative float (default 1)
        Resolution parameter for the modularity matrix. A smaller value assigns
        maximum modularity to partitions with large communities, while a larger
        value assigns maximum modularity to partitions with many small
        communities.
    null : callable (default `girvan_newman_null`)
        Function of the input tensor block that returns, for each adjacency
        matrix in the input tensor block, a suitable null model.
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
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
    **params
        Any additional parameters are passed to the null model.
    """
    def __init__(self, nu, affiliation_xfm=None, exclude_diag=True, gamma=1,
                 null=girvan_newman_null, normalise_modularity=True,
                 normalise_coaffiliation=True, directed=False, sign='+',
                 name=None, **params):
        super(ModularityLoss, self).__init__(nu=nu)
        if affiliation_xfm is None:
            affiliation_xfm = lambda x: x
        self.affiliation_xfm = affiliation_xfm
        self.exclude_diag = exclude_diag
        self.gamma = gamma
        self.null = null
        self.normalise_modularity = normalise_modularity
        self.normalise_coaffiliation = normalise_coaffiliation
        self.directed = directed
        self.sign = sign
        self.name = name
        self.params = params

    def forward(self, A, C, C_o=None, L=None):
        r"""
        Compute a differentiable relaxation of the Girvan-Newman modularity.

        Parameters
        ----------
        A : Tensor
            Block of adjacency matrices for which the modularity is to be
            computed.
        C : Tensor
            Proposed community affiliation of vertices in the source set. Each
            slice is a matrix :math:`C^{(i)} \in \mathbb{R}^{I \ times C}`
            that encodes the uncertainty in each vertex's community
            assignment. :math:`C^{(i)}_{jk}` denotes the probability that
            vertex j is assigned to community k. If this is binary-valued,
            then it reflects a deterministic assignment and reduces to the
            standard Girvan-Newman modularity.
        C_o : Tensor or None (default None)
            Community affiliation of vertices in the sink set. If None, then
            it is assumed that the source and sink sets are the same, and
            `C_o` is set equal to `C`.
        L : Tensor or None (default None)
            Probability of affiliation between communities. Each entry
            :math:`L_{ij}` encodes the probability of a vertex in community i
            connecting with a vertex in community j. If None, then a strictly
            assortative structure is assumed (equivalent to L equals
            identity), under which nodes in the same community preferentially
            coaffiliate while nodes in different communities remain
            disaffiliated.
        """
        if C_o is None:
            C_o = C
        out = -self.nu * relaxed_modularity(
            A=A,
            C=self.affiliation_xfm(C),
            C_o=self.affiliation_xfm(C_o),
            L=L,
            exclude_diag=self.exclude_diag,
            gamma=self.gamma,
            null=self.null,
            normalise_modularity=self.normalise_modularity,
            normalise_coaffiliation=self.normalise_coaffiliation,
            directed=self.directed,
            sign=self.sign,
            **self.params
        ).squeeze()
        message = {
            'NAME': self.name,
            'LOSS': out.clone().detach().item(),
            'NU': self.nu
        }
        for s in self.listeners:
            s._listen(message)
        return out
