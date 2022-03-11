# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Reference proximity
~~~~~~~~~~~~~~~~~~~
Regularisations using centre-of-mass proximity to a reference.
"""
from functools import partial
from .norm import NormedRegularisation, ReducingRegularisation
from ..functional.cmass import cmass_reference_displacement, diffuse


class CentroidAnchor(NormedRegularisation):
    def __init__(self, refs, nu=1, axes=None, na_rm=False, norm=2):
        reg = partial(
            cmass_reference_displacement,
            refs=refs,
            axes=axes,
            na_rm=na_rm
        )
        super(SymmetricBimodal, self).__init__(nu=nu, p=norm, reg=reg)


class Compactness(ReducingRegularisation):
    def __init__(self, coor, nu=1, norm=2, floor=0,
                 radius=None, reduction=None):
        reduction = reduction or torch.mean
        reg = partial(
            diffuse,
            coor=coor,
            norm=norm,
            floor=floor,
            radius=radius
        )
        super(Compactness, self).__init__(
            nu=nu,
            reduction=reduction,
            reg=reg
        )
