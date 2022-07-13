# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Connectopic Loss Functional
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Basic, minimal implementation of the connectopic loss functional.
"""
import torch
from functools import partial
from .base import ReducingLoss
from ..functional.kernel import linear_distance


def connectopy_loss(Q, A, dissimilarity=None, affinity=None,
                    D=None, theta=None, omega=None):
    r"""
    Connectopy loss, for computing different kinds of connectopic maps.

    .. admonition:: Connectopic loss functional

        Given an affinity matrix A, the connectopic loss minimises the objective

        :math:`\mathbf{1}^\intercal \left( \mathbf{A} \circ S_\theta(\mathbf{Q}) \right) \mathbf{1}`

        for a pairwise function S. The default pairwise function is the square
        of the L2 distance. The columns of the Q that minimises the objective
        are the learned connectopic maps.

    .. warning::
        If you're using this for a well-characterised connectopic map with a
        closed-form or algorithmically optimised solution, such as Laplacian
        eigenmaps or many forms of community detection, then in most cases you
        would be better off directly computing exact maps rather than using this
        loss functional to approximate them.

        Because this operation attempts to learn all of the maps that jointly
        minimise the objective in a single shot rather than using iterative
        projection, it is more prone to misalignment than a projective approach
        for eigendecomposition-based maps.

    .. danger::
        Note that the ``connectopy_loss`` is often insufficient on its own. It
        should be combined with appropriate constraints, for instance to ensure
        the learned maps are zero-centred and orthogonal.

    :Dimension: **Q :** :math:`(D, C)`
                    D denotes the number of vertices in the affinity matrix
                    and C denotes the number of proposed maps.
                **A :** :math:`(D, D)`
                    As above.
                **D :** :math:`(D, D)`
                    As above.
                **theta :** :math:`(C)` or :math:`(C, C)`
                    As above.

    Parameters
    ----------
    Q : tensor
        Proposed connectopies or maps.
    A : tensor
        Affinity matrix.
    dissimilarity : callable
        Function to compute dissimilarity between latent coordinates induced
        by the proposed connectopies. By default, the square of the L2
        distance is used. The callable must accept ``Q`` and ``theta`` as
        arguments. (``theta`` may be unused.)
    affinity : callable or None (default None)
        If an affinity function is provided, then the image of argument A
        under this function is the affinity matrix. Otherwise, argument A is
        the affinity matrix.
    D : tensor or None (default None)
        If this argument is provided, then the affinity matrix is first
        transformed as :math:`D A D^\intercal`. For instance, setting D to
        a diagonal matrix whose entries are the reciprocal of the square root
        of vertex degrees corresponds to learning eigenmaps of a normalised
        graph Laplacian.
    theta : tensor, float, or None (default None)
        Scaling factors for the columns in :math:`Q` when the loss is
        computed. By default, the last column has a weight of 1, the
        second-to-last has a weight of 2, and so on. This is used to encourage
        the last column to correspond to the least important eigenmap and the
        first column to correspond to the most important eigenmap.
    omega : tensor, float, or None (default None)
        Optional parameterisation of the affinity function, if one is
        provided.
    """
    if theta is None:
        n_vecs = Q.size(-1)
        theta = torch.arange(n_vecs, 0, -1, dtype=Q.dtype, device=Q.device)
    if dissimilarity is None:
        dissimilarity = lambda Q, theta: linear_distance(Q, theta=theta)
    if affinity is not None:
        A = affinity(A, omega=omega)
    if D is not None:
        A = D @ A @ D.t()
    return (dissimilarity(Q, theta) * A).sum((-2, -1))


class Connectopy(ReducingLoss):
    def __init__(self, dissimilarity, affinity=None,
                 nu=1, reduction=None, name=None):
        if reduction is None:
            reduction = torch.mean
        self.dissimilarity = dissimilarity
        self.affinity = affinity
        loss = partial(
            connectopy_loss,
            dissimilarity=dissimilarity,
            affinity=affinity
        )
        super().__init__(nu=nu, reduction=reduction, loss=loss, name=name)
