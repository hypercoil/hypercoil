# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Brain surface plots with `surfplot`.
These currently require a custom patch of `surfplot`.
"""
import surfplot
import torch
import nibabel as nb
import templateflow.api as tflow
from hypercoil.neuro.const import fsLR
from hypercoil.viz.surfutils import _CMapFromSurfMixin


class fsLRAtlasParcels(_CMapFromSurfMixin):
    def __init__(self, atlas):
        self.atlas = atlas
        coor_query = fsLR().TFLOW_COOR_QUERY
        coor_query.update(suffix='veryinflated')
        self.lh, self.rh = (
            tflow.get(**coor_query, hemi='L'),
            tflow.get(**coor_query, hemi='R')
        )
        self.dim_lh = nb.load(self.lh).darrays[0].dims[0]
        self.dim_rh = nb.load(self.rh).darrays[0].dims[0]
        self.dim = self.dim_lh + self.dim_rh
        self.data_mask = torch.ones_like(self.atlas.mask)
        self.cmap_mask = self.atlas.mask.clone()
        self.data_mask[self.dim:] = False
        self.cmap_mask[self.dim:] = False
        self.cortical_mask = (
            self.atlas.compartments['cortex_L'] |
            self.atlas.compartments['cortex_R']
        )
        self.coor = self.atlas.coors[self.cortical_mask[self.atlas.mask]]

    def drop_null(self, null=0):
        subset = (self.data != null)
        return self.data[subset], self.coor[subset]

    def __call__(self, cmap):
        data = torch.zeros(len(self.atlas.mask), dtype=torch.long)
        for compartment in ('cortex_L', 'cortex_R'):
            mask = self.atlas.compartments[compartment]
            labels = self.atlas.maps[compartment].argmax(0)
            compartment_data = (
                self.atlas.decoder[compartment][labels]
            )
            compartment_data[self.atlas.maps[compartment].sum(0) == 0] = 0
            data[mask] = compartment_data
        # We're assuming cortex-first ordering in the atlas mask.
        self.data = data[self.cmap_mask]
        cmap = self._select_cmap(cmap=cmap)
        self.data = data[self.data_mask].numpy()

        p = surfplot.Plot(
            surf_lh=self.lh,
            surf_rh=self.rh,
            size=(200, 300),
            zoom=3,
            views='dorsal',
            brightness=1
        )
        p.add_layer(
            self.data.astype('long')[:self.dim],
            cmap=cmap,
            color_range=(1, len(cmap.colors)),
            cbar=None
        )
        fig = p.build()
        fig.set_dpi(200)
        fig.show()
