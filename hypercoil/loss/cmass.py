# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Reference proximity
~~~~~~~~~~~~~~~~~~~
Loss functions using spatial proximity to a reference centre-of-mass.
"""
from functools import partial
from torch import mean
from .base import ReducingLoss
from .norm import NormedLoss
from ..functional.cmass import (
    cmass_coor,
    cmass_reference_displacement,
    diffuse
)


def interhemispheric_anchor(lh, rh, lh_coor, rh_coor, radius=100):
    cm_lh = cmass_coor(X=lh, coor=lh_coor, radius=radius)
    cm_lh[0, :] = -cm_lh[0, :]
    return cmass_reference_displacement(
        weight=rh,
        refs=cm_lh,
        coor=rh_coor,
        radius=radius
    )


class HemisphericTether(NormedLoss):
    r"""
    Displacement of centres of mass in one cortical hemisphere from
    corresponding centres of mass in the other cortical hemisphere.

    .. admonition:: Hemispheric Tether

        The hemispheric tether is defined as

        :math:`\sum_{\ell} \left\| \ell_{LH, centre} - \ell_{RH, centre} \right\|`

        where :math:`\ell` denotes a pair of regions, one in each cortical
        hemisphere.

        .. image:: ../_images/spatialnull.gif
            :width: 500
            :align: center

        `The symmetry of this spatial null model is enforced through a
        moderately strong hemispheric tether.`

    When an atlas is initialised with the same number of parcels in each
    cortical hemisphere compartment, the hemispheric tether can be used to
    approximately enforce symmetry and to enforce analogy between a pair of
    parcels in the two cortical hemispheres.

    .. warning::
        Currently, this loss only works in spherical surface space.
    """
    def __init__(self, nu=1, radius=100, axis=-2, norm=2,
                 reduction=None, name=None):
        loss = partial(interhemispheric_anchor, radius=radius)
        super(HemisphericTether, self).__init__(
            nu=nu, p=norm, precompose=loss, axis=axis,
            reduction=reduction, name=name)


class CentroidAnchor(NormedLoss):
    """
    Displacement of centres of mass from reference points.
    """
    def __init__(self, refs, nu=1, axis=-2, reduction=None,
                 radius=None, norm=2, name=None):
        loss = partial(
            cmass_reference_displacement,
            refs=refs,
            radius=radius
        )
        super(CentroidAnchor, self).__init__(
            nu=nu, p=norm, precompose=loss, axis=axis,
            reduction=reduction, name=name)


class Compactness(ReducingLoss):
    r"""
    Compactness score for a coordinate/weight pair.

    .. admonition:: Compactness

        The compactness is defined as

        :math:`\mathbf{1}^\intercal\left(A \circ \left\|C - \frac{AC}{A\mathbf{1}} \right\|_{cols} \right)\mathbf{1}`

        Given a coordinate set :math:`C` for the columns of a weight
        :math:`A`, the compactness measures the weighted average norm of the
        displacement of each of the weight's entries from its row's centre of
        mass. (The centre of mass is expressed above as
        :math:`\frac{C \circ A}{\mathbf{1}^\intercal A}`).

        .. image:: ../_images/compactloss.gif
            :width: 200
            :align: center

        `In this simulation, the compactness loss is applied with a
        multi-logit domain mapper and without any other losses or
        regularisations. The weights collapse to compact but unstructured
        regions of the field.`

    Penalising this quantity can promote more compact rows (i.e., concentrate
    the weight in each row over columns corresponding to coordinates close to
    the row's spatial centre of mass).

    .. warning::
        This loss can have a large memory footprint, because it requires
        computing an intermediate tensor with dimensions equal to the number
        of rows in the weight, multiplied by the number of columns in the
        weight, multiplied by the dimension of the coordinate space.

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
    reduction : callable (default ``torch.mean``)
        Map from a tensor of arbitrary dimension to a scalar. The output of
        the compactness loss is passed into ``reduction`` to return a scalar.
    name : str or None (default None)
        Identifying string for the instantiation of the loss object.
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
