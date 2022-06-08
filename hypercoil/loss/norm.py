# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Normed penalty
~~~~~~~~~~~~~~
Generalised module for applying a normed penalty to the weight parameter
of a module. Note that this is currently configured to use vector norms
(so the 2-norm will be the Frobenius norm for a matrix rather than its
largest singular value).
"""
import torch
from torch.linalg import vector_norm as pnorm
from torch.nn import Module
from functools import partial
from .base import ReducingLoss


def norm_reduction(X, p=2, axis=-1, reduction=None):
    """
    Compute a specified norm along an axis or set of axes, and then map the
    tensor of norms to a scalar using a reduction map.
    """
    reduction = reduction or torch.mean
    norm = pnorm(X, ord=p, dim=axis)
    return reduction(norm)


class NormedLoss(ReducingLoss):
    """
    Generalised module for computing losses based on the norm of a tensor.

    .. note::
        Note that this is currently configured to use vector norms (so the
        2-norm will be the Frobenius norm for a matrix rather than its
        largest singular value).

    Parameters
    ----------
    nu : float
        Loss function weight multiplier.
    p : float (default 2)
        Norm order. ``p=1`` corresponds to the Manhattan L1 norm, ``p=2``
        corresponds to the Euclidean L2 norm, etc.
    precompose : callable or None (default None)
        Pre-transformation of the input tensor, on whose output the norm is
        computed.
    axis : int, iterable(int), or None (default None)
        Axes defining the slice of the input tensor over which the norm is
        computed. If this is None, then the overall tensor norm is computed.
    reduction : callable (default ``torch.mean``)
        Map from a tensor of arbitrary dimension to a scalar. The output of
        the norm operation over the specified axes produces a tensor whose
        extent over remaining axes is unreduced. This output tensor is then
        passed into ``reduction`` to return a scalar.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
    """
    def __init__(self, nu, p=2, precompose=None, axis=None,
                 reduction=None, name=None):
        reduction = partial(
            norm_reduction,
            p=p,
            axis=axis,
            reduction=reduction
        )
        if precompose is None:
            precompose = lambda x: x
        super(NormedLoss, self).__init__(
            nu=nu, reduction=reduction, loss=precompose, name=name
        )
        self.p = p

    def extra_repr(self):
        return [f'norm=L{self.p}']


class UnilateralNormedLoss(NormedLoss):
    """
    Unilateral version of :class:`NormedLoss`.

    Any nonpositive weights are not considered in norm computation. To exclude
    weights in a set other than nonpositive numbers from the computation, pass
    a map from that set into the nonpositive numbers to ``precompose``.

    Parameters
    ----------
    nu : float
        Loss function weight multiplier.
    p : float (default 2)
        Norm order. ``p=1`` corresponds to the Manhattan L1 norm, ``p=2``
        corresponds to the Euclidean L2 norm, etc.
    precompose : callable or None (default None)
        Pre-transformation of the input tensor, on whose output the unilateral
        norm is computed.
    axis : int, iterable(int), or None (default None)
        Axes defining the slice of the input tensor over which the norm is
        computed. If this is None, then the overall tensor norm is computed.
    reduction : callable (default ``torch.mean``)
        Map from a tensor of arbitrary dimension to a scalar. The output of
        the norm operation over the specified axes produces a tensor whose
        extent over remaining axes is unreduced. This output tensor is then
        passed into ``reduction`` to return a scalar.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
    """
    def __init__(self, nu, p=2, precompose=None, axis=None,
                 reduction=None, name=None):
        if precompose is None:
            precompose = lambda x: x
        f = lambda x: torch.maximum(
            precompose(x),
            torch.tensor(0, dtype=x.dtype, device=x.device)
        )
        super(UnilateralNormedLoss, self).__init__(
            nu=nu, p=p, precompose=f, axis=axis,
            reduction=reduction, name=name
        )
