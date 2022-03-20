# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Reference proximity
~~~~~~~~~~~~~~~~~~~
Loss functions using centre-of-mass proximity to a reference.
"""
from functools import partial
from torch import mean
from .base import ReducingLoss
from .norm import NormedLoss
from ..functional.cmass import cmass_reference_displacement, diffuse


#TODO: rework this to use cmass_coor instead of cmass in functional.
class CentroidAnchor(NormedLoss):
    """
    Displacement of centres of mass from reference points.
    """
    def __init__(self, refs, nu=1, axes=None, na_rm=False,
                 norm=2, name=None):
        loss = partial(
            cmass_reference_displacement,
            refs=refs,
            axes=axes,
            na_rm=na_rm
        )
        super(CentroidAnchor, self).__init__(
            nu=nu, p=norm, loss=loss, name=name)


class Compactness(ReducingLoss):
    r"""
    Compute compactness score for a coordinate/weight pair.

    The compactness is defined as

    :math:`\mathbf{1}^\intercal\left(A \circ \left\|C - \frac{C \circ A}{\mathbf{1}^\intercal A} \right\|_{cols} \right)\mathbf{1}`

    Given a coordinate system :math:`C` for the columns of a weight :math:`A`,
    the compactness measures the weighted average norm of the displacement of
    each of the weight's entries from its row's centre of mass.

    Penalising this quantity can promote more compact rows (i.e., concentrate
    the weight in each row over columns corresponding to coordinates close to
    the row's spatial centre of mass).

    Parameters
    ----------
    coor : tensor
        Coordinates tensor. Each column of the coordinates tensor should
        determine the coordinates associated to the corresponding column of
        the input tensor during the forward pass. The number of rows
        accordingly corresponds to the dimension of the coordinate space.
    nu : float (default 1)
        Loss function weight multiplier.
    norm : float (default 2)
        Designation for the p-norm used to operationalise the dispersion of
        a weight around its centre of mass.
    floor : float (default 0)
        Maximum unpenalised distance from the centre of mass. If this is a
        nonzero value d, then weights within a norm ball of radius d receive
        no penalty, and the penalty is instead applied in proportion to
        distance from this norm ball.
    radius : nonnegative float or None (default None)
        If this is a nonnegative float, then the distances are computed as a
        spherical geodesic instead of a p-norm. In this case, the centre of
        mass is operationalised as the Euclidean centre of mass, projected
        onto the surface of a sphere of the specified radius. The coordinate
        system must be three-dimensional, and each coordinate should then
        correspond to a point on the surface of the sphere of the specified
        radius.
    reduction : callable (default None)
        Map from a tensor of arbitrary dimension to a scalar. The output of
        `loss` is passed into `reduction` to return a scalar.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.

    Notes
    -----
        This loss can have a large memory footprint, because it requires
        computing an intermediate tensor with dimensions equal to the number
        of rows in the weight, multiplied by the number of columns in the
        weight, multiplied by the dimension of the coordinate space.
    """
    def __init__(self, coor, nu=1, norm=2, floor=0,
                 radius=None, reduction=None, name=None):
        reduction = reduction or mean
        loss = partial(
            diffuse,
            coor=coor,
            norm=norm,
            floor=floor,
            radius=radius
        )
        super(Compactness, self).__init__(
            nu=nu,
            reduction=reduction,
            loss=loss,
            name=name
        )
