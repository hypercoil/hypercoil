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
    to standard behaviour when the inputs it is provided are standard (i.e.,
    deterministic and associative).

    .. admonition:: Girvan-Newman Modularity Relaxation

        The relaxed modularity loss is defined as the negative sum of all
        entries in the Hadamard (elementwise) product between the modularity
        matrix and the coaffiliation matrix.

        :math:`\mathcal{L}_Q = -\nu_Q \mathbf{1}^\intercal \left( B \circ H \right) \mathbf{1}`

        .. image:: ../_images/modularityloss.svg
            :width: 500
            :align: center

        - The modularity matrix :math:`B` is the difference between the
          observed connectivity matrix :math:`A` and the expected connectivity
          matrix under some null model that assumes no community structure,
          :math:`P`: :math:`B = A - \gamma P`.

          - The community resolution parameter :math:`\gamma` essentially
            determines the scale of the community structure that optimises the
            relaxed modularity loss.

          - By default, we use the Girvan-Newman null model
            :math:`P_{GN} = \frac{A \mathbf{1} \mathbf{1}^\intercal A}{\mathbf{1}^\intercal A \mathbf{1}}`,
            which can be interpreted as the expected weight of connections
            between each pair of vertices if all existing edges are cut and
            then randomly rewired. 

          - Note that :math:`A \mathbf{1}` is the in-degree of the adjacency
            matrix and :math:`\mathbf{1}^\intercal A` is its out-degree, and
            the two are transposes of one another for symmetric :math:`A`.
            Also note that the denominator
            :math:`\mathbf{1}^\intercal A \mathbf{1}` is twice the number of
            edges for an undirected graph.)

        - The coaffiliation matrix :math:`H` is calculated as
          :math:`H = C_{in} L C_{out}^\intercal`, where
          :math:`C_{in} \in \mathbb{R}^{(v_{in} \times c)}` and
          :math:`C_{out} \in \mathbb{R}^{(v_{out} \times c)}` are proposed
          assignment weights of in-vertices and out-vertices to communities.
          :math:`L \in \mathbb{R}^{c \times c)}` is the proposed coupling
          matrix among each pair of communities and defaults to identity to
          represent associative structure.

          - Note that, when :math:`C_{in} = C_{out}` is deterministically in
            :math:`\{0, 1\}` and :math:`L = I`, this term reduces to the
            familiar delta-function notation for the true Girvan-Newman
            modularity.

    Penalising this favours a weight that induces a modular community
    structure on the input matrix -- or, an input matrix whose structure
    is reasonably accounted for by the proposed community affiliation
    weights.

    .. warning::
        To conform with the network community interpretation of this loss
        function, parameters representing the community affiliation :math:`C`
        and coupling :math:`L` matrices can be pre-transformed. Mapping the
        community affiliation matrix :math:`C` through a
        :doc:`softmax <hypercoil.functional.domain.MultiLogit>`
        function along the community axis lends the affiliation matrix the
        intuitive interpretation of distributions over communities, or a
        quantification of the uncertainty of each vertex's community
        assignment. Similarly, the coupling matrix can be pre-transformed
        through a
        :doc:`sigmoid <hypercoil.functional.domain.Logit>`
        to constrain inter-community couplings to :math:`(0, 1)`.
    .. note::
        Because the community affiliation matrices :math:`C` induce
        parcellations, we can regularise them using parcellation losses. For
        instance, penalising the
        :doc:`entropy <hypercoil.loss.entropy>`
        will promote a solution wherein each node's community assignment
        probability distribution is concentrated in a single community.
        Similarly, using parcel
        :doc:`equilibrium <hypercoil.loss.equilibrium>` will favour a solution
        wherein communities are of similar sizes.

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
                 reduction=None, name=None, **params):
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
        self.reduction = reduction or torch.mean

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
        out = self.reduction(out)
        message = {
            'NAME': self.name,
            'LOSS': out.clone().detach().item(),
            'NU': self.nu
        }
        for s in self.listeners:
            s._listen(message)
        return out
