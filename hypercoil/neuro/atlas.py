# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlases
~~~~~~~
Standard brain atlases.
"""
import templateflow.api as tflow
from ..init.atlas import CortexSubcortexCIfTIAtlas
from ..neuro.const import fsLR


class fsLRAtlas(CortexSubcortexCIfTIAtlas):
    def __init__(self, ref_pointer, name=None, clear_cache=True,
                 dtype=None, device=None):
        surf_L = tflow.get(**fsLR.TFLOW_COOR_QUERY,
                           **fsLR.TFLOW_COMPARTMENTS['L'])
        surf_R = tflow.get(**fsLR.TFLOW_COOR_QUERY,
                           **fsLR.TFLOW_COMPARTMENTS['R'])
        mask_L = tflow.get(**fsLR.TFLOW_MASK_QUERY,
                           **fsLR.TFLOW_COMPARTMENTS['L'])
        mask_R = tflow.get(**fsLR.TFLOW_MASK_QUERY,
                           **fsLR.TFLOW_COMPARTMENTS['R'])
        super(fsLRAtlas, self).__init__(
            ref_pointer=ref_pointer,
            surf_L=surf_L, surf_R=surf_R,
            mask_L=mask_L, mask_R=mask_R, name=name,
            cortex_L='CIFTI_STRUCTURE_CORTEX_LEFT',
            cortex_R='CIFTI_STRUCTURE_CORTEX_RIGHT',
            dtype=dtype, device=device
        )


#TODO: add the Gordon and Glasser surface atlases here as soon as they're
# registered into TemplateFlow.

#TODO: add all atlases from TemplateFlow.
