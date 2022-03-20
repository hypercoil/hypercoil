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


class CentroidAnchor(NormedLoss):
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
