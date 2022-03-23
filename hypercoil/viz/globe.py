# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Brain globe plots.
"""
import torch
import numpy as np
import templateflow.api as tflow
import matplotlib.pyplot as plt
from hypercoil.functional.sphere import sphere_to_latlong
from hypercoil.neuro.const import fsLR
from hypercoil.viz.surfutils import (
    data_from_struc_tag,
    _SurfFromFilesMixin,
    _SurfNoActionMixin,
    _CMapFromSurfMixin,
    _CMapNoActionMixin
)


def brain_globe(data, coor, shift=0, cmap='flag', figsize=(12, 12),
                projection='mollweide', dtype=None, device=None):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=dtype, device=device)
    if not isinstance(coor, torch.Tensor):
        coor = torch.tensor(coor, dtype=data.dtype, device=data.device)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection)

    try:
        lat, lon = coor.clone().T
    except ValueError:
        lat, lon = sphere_to_latlong(coor).T
    lon += shift
    lon[lon < -np.pi] += (2 * np.pi)
    lon[lon > np.pi] -= (2 * np.pi)

    col = data.squeeze()

    order = np.argsort(col.squeeze())

    ax.scatter(lon[order], lat[order], c=col[order], cmap=cmap, marker='.')

    ax.set_facecolor('black')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    return ax


class _GlobeBrain:
    def __init__(self, data, coor=None, shift=0,
                 coor_mask=None, struc_tag=None, cmap='flag',
                 figsize=(12, 12), projection='mollweide',
                 dtype=None, device=None):
        self.data, self.coor = self._select_data_and_coor(
            data, coor,
            coor_mask,
            struc_tag
        )
        self.coor = self._transform_coor(
            self.coor,
            dtype=dtype,
            device=device)
        self.cmap = self._select_cmap(cmap, struc_tag)
        self.shift = shift
        self.figsize = figsize
        self.projection = projection

    def drop_null(self, null=0):
        subset = (self.data != null)
        return self.data[subset], self.coor[subset]

    def _transform_coor(self, coor, dtype, device):
        coor = sphere_to_latlong(
            torch.tensor(coor, dtype=dtype, device=device))
        return coor

    def __call__(self):
        return brain_globe(
            data=self.data, coor=self.coor,
            shift=self.shift, cmap=self.cmap,
            figsize=self.figsize, projection=self.projection
        )


class _GlobeCortexL(_GlobeBrain):
    def __init__(self, data, coor=None, cmap='flag',
                 figsize=(12, 12), projection='mollweide'):
        super().__init__(
            data=data, coor=coor, shift=(3 * np.pi / 4),
            cmap=cmap, figsize=figsize, projection=projection
        )


class _GlobeCortexR(_GlobeBrain):
    def __init__(self, data, coor=None, cmap='flag',
                 figsize=(12, 12), projection='mollweide'):
        super().__init__(
            data=data, coor=coor, shift=(np.pi / 4),
            cmap=cmap, figsize=figsize, projection=projection
        )


class _CortexLfsLR32KMixin(_SurfFromFilesMixin):
    def _select_data_and_coor(self, data, coor,
                              coor_mask=None,
                              struc_tag=None):
        return super()._select_data_and_coor(
            data=data,
            coor=tflow.get(**fsLR.TFLOW_COOR_QUERY,
                           **fsLR.TFLOW_COMPARTMENTS['L']),
            coor_mask=tflow.get(**fsLR.TFLOW_MASK_QUERY,
                                **fsLR.TFLOW_COMPARTMENTS['L']),
            struc_tag='CIFTI_STRUCTURE_CORTEX_LEFT'
        )

    def _select_cmap(self, cmap, struc_tag=None):
        return super()._select_cmap(
            cmap=cmap, struc_tag='CIFTI_STRUCTURE_CORTEX_LEFT')


class _CortexRfsLR32KMixin(_SurfFromFilesMixin):
    def _select_data_and_coor(self, data, coor,
                              coor_mask=None,
                              struc_tag=None):
        return super()._select_data_and_coor(
            data=data,
            coor=tflow.get(**fsLR.TFLOW_COOR_QUERY,
                           **fsLR.TFLOW_COMPARTMENTS['R']),
            coor_mask=tflow.get(**fsLR.TFLOW_MASK_QUERY,
                                **fsLR.TFLOW_COMPARTMENTS['R']),
            struc_tag='CIFTI_STRUCTURE_CORTEX_RIGHT'
        )

    def _select_cmap(self, cmap, struc_tag=None):
        return super()._select_cmap(
            cmap=cmap, struc_tag='CIFTI_STRUCTURE_CORTEX_RIGHT')


class _ModalGlobeMixin:
    def _config_map_and_cmap(self):
        pass


class _NetworkGlobeMixin:
    def _config_map_and_cmap(self):
        pass


class GlobeBrain(
    _GlobeBrain,
    _SurfNoActionMixin,
    _CMapNoActionMixin
):
    pass


class GlobeFromFiles(
    _GlobeBrain,
    _SurfFromFilesMixin,
    _CMapNoActionMixin
):
    pass


class CortexLfsLRFromFiles(
    _GlobeCortexL,
    _CortexLfsLR32KMixin,
    _CMapFromSurfMixin
):
    pass


class CortexRfsLRFromFiles(
    _GlobeCortexR,
    _CortexRfsLR32KMixin,
    _CMapFromSurfMixin
):
    pass
