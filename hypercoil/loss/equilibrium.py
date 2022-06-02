# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
r"""
Loss functions to favour equal weight across one dimension of a parcellation
tensor.

.. admonition:: Equilibrium

    The equilibrium loss of a parcellation tensor :math:`A` is defined as

    :math:`\mathbf{1}^\intercal \left[\left(A \mathbf{1}\right) \circ \left(A \mathbf{1}\right) \right]`

    The parcel equilibrium is principally designed to operate on parcellation
    tensors. A parcellation tensor is one whose rows correspond to features
    (e.g., voxels, time points, frequency bins, or network nodes) and whose
    columns correspond to parcels. Element :math:`i, j` in this tensor
    accordingly indexes the assignment of feature :math:`j` to parcel
    :math:`i`. Examples of parcellation tensors might include atlases that map
    voxels to regions or affiliation matrices that map graph vertices to
    communities. It is often desirable to constrain feature-parcel assignments
    to :math:`[0, k]` for some :math:`k` and ensure that the sum over each
    feature's assignment is always :math:`k`. (Otherwise, the unnormalised
    loss could be improved by simply shrinking all weights.) For this reason,
    we can either normalise the loss or situate the parcellation tensor in the
    probability simplex using a
    :doc:`multi-logit (softmax) domain mapper <hypercoil.functional.domain.MultiLogit>`.

    The parcel equilibrium attains a minimum when parcels are equal in their
    total weight. It has a trivial and uninteresting minimum where all parcel
    assignments are equiprobable for all features. Other minima, which might
    be of greater interest, occur where each feature is deterministically
    assigned to a single parcel. These minima can be favoured by using the
    equilibrium in conjunction with a penalty on the
    :doc:`entropy <hypercoil.loss.entropy>`.
"""
import torch
from functools import partial
from .base import ReducingLoss


def equilibrium(X, axis=-1):
    """
    Compute the parcel equilibrium.
    """
    return X.mean(axis) ** 2


def softmax_equilibrium(X, axis=-1, prob_axis=-2):
    """
    Project logits in the input matrix onto the probability simplex, and then
    compute the parcel equilibrium.
    """
    probs = torch.softmax(X, axis=prob_axis)
    return equilibrium(probs, axis=axis)


class Equilibrium(ReducingLoss):
    """
    Parcel equilibrium.

    The parcel equilibrium is principally designed to operate on parcellation
    tensors. A parcellation tensor is one whose columns correspond to features
    (e.g., voxels, time points, frequency bins, or network nodes) and whose
    rows correspond to parcels. Element i, j in this tensor accordingly
    indexes the assignment of feature j to parcel i. Examples of parcellation
    tensors might include atlases that map voxels to regions or affiliation
    matrices that map graph vertices to communities. It is often desirable
    to constrain feature-parcel assignments to [0, k] for some k and ensure
    that the sum over each feature's assignment is always k. For this reason,
    we can situate the parcellation tensor in the probability simplex using
    a multi-logit (softmax) domain mapper.

    The parcel equilibrium attains a minimum when parcels are equal in their
    total weight. It has a trivial and uninteresting minimum where all parcel
    assignments are equiprobable for all features. Other minima, which might
    be of greater interest, occur where each feature is deterministically
    assigned to a single parcel. These minima can be favoured by using the
    equilibrium in conjunction with a penalty on the entropy.

    Parameters
    ----------
    nu : float (default 1)
        Loss function weight multiplier.
    axis : int (default -2)
        Vectors along the specified axis should correspond to the assignment
        probabilities of every feature to a single parcel.
    reduction : callable (default `torch.mean`)
        Map from a tensor of arbitrary dimension to a scalar. This has no
        effect unless the input is a batch or block of parcellation tensors,
        in which case it reduces over the batch.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
    """
    def __init__(self, nu=1, axis=-1, reduction=None, name=None):
        reduction = reduction or torch.mean
        loss = partial(equilibrium, axis=axis)
        super(Equilibrium, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=loss,
            name=name
        )


class SoftmaxEquilibrium(ReducingLoss):
    """
    Parcel equilibrium, precomposed with a softmax function.

    The parcel equilibrium is principally designed to operate on parcellation
    tensors. A parcellation tensor is one whose columns correspond to features
    (e.g., voxels, time points, frequency bins, or network nodes) and whose
    rows correspond to parcels. Element i, j in this tensor accordingly
    indexes the assignment of feature j to parcel i. Examples of parcellation
    tensors might include atlases that map voxels to regions or affiliation
    matrices that map graph vertices to communities. It is often desirable
    to constrain feature-parcel assignments to [0, k] for some k and ensure
    that the sum over each feature's assignment is always k. For this reason,
    we can situate the parcellation tensor in the probability simplex using
    a multi-logit (softmax) domain mapper. This version of the equilibrium
    loss operates in this manner and accordingly expects logit inputs.
    Use ``Equilibrium`` instead if your inputs will already contain
    probabilities.

    The parcel equilibrium attains a minimum when parcels are equal in their
    total weight. It has a trivial and uninteresting minimum where all parcel
    assignments are equiprobable for all features. Other minima, which might
    be of greater interest, occur where each feature is deterministically
    assigned to a single parcel. These minima can be favoured by using the
    equilibrium in conjunction with a penalty on the entropy.

    Parameters
    ----------
    nu : float (default 1)
        Loss function weight multiplier.
    axis : int (default -2)
        Vectors along the specified axis should correspond to the assignment
        probabilities of every feature to a single parcel.
    reduction : callable (default `torch.mean`)
        Map from a tensor of arbitrary dimension to a scalar. This has no
        effect unless the input is a batch or block of parcellation tensors,
        in which case it reduces over the batch.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
    """
    def __init__(self, nu=1, axis=-1, prob_axis=-2,
                 reduction=None, name=None):
        reduction = reduction or torch.mean
        loss = partial(softmax_equilibrium, axis=axis, prob_axis=prob_axis)
        super(SoftmaxEquilibrium, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=loss,
            name=name
        )
