# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlases
~~~~~~~
Standard brain atlases.
"""
import templateflow.api as tflow
from ..init.atlas import SurfaceAtlas
from ..neuro.const import fsLR


class fsLRAtlas(SurfaceAtlas):
    def __init__(self, path, name=None, label_dict=None, null=0,
                 max_bin=10000, truncate=None, spherical_scale=1.):
        surf_L = tflow.get(**fsLR.TFLOW_COOR_QUERY,
                           **fsLR.TFLOW_COMPARTMENTS['L'])
        surf_R = tflow.get(**fsLR.TFLOW_COOR_QUERY,
                           **fsLR.TFLOW_COMPARTMENTS['R'])
        mask_L = tflow.get(**fsLR.TFLOW_MASK_QUERY,
                           **fsLR.TFLOW_COMPARTMENTS['L'])
        mask_R = tflow.get(**fsLR.TFLOW_MASK_QUERY,
                           **fsLR.TFLOW_COMPARTMENTS['R'])
        super(fsLRAtlas, self).__init__(
            path=path, surf_L=surf_L, surf_R=surf_R, name=name,
            label_dict=label_dict, mask_L=mask_L, mask_R=mask_R, null=null,
            cortex_L='CIFTI_STRUCTURE_CORTEX_LEFT',
            cortex_R='CIFTI_STRUCTURE_CORTEX_RIGHT', max_bin=max_bin,
            truncate=truncate, spherical_scale=spherical_scale
        )


#TODO: add the Gordon and Glasser surface atlases here as soon as they're
# registered into TemplateFlow.

#TODO: add all atlases from TemplateFlow.
