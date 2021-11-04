# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Brain globe plots.
"""
import torch
import numpy as np
import nibabel as nb
import templateflow.api as tflow
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from hypercoil.functional.sphere import sphere_to_latlong


TFLOW_MASK_QUERY = {
    'template': 'fsLR',
    'density': '32k',
    'desc': 'nomedialwall'
}
TFLOW_COOR_QUERY = {
    'template': 'fsLR',
    'space': None,
    'density': '32k',
    'suffix': 'sphere'
}
TFLOW_COMPARTMENTS = {
    'L': {'hemi': 'L'},
    'R': {'hemi': 'R'}
}


def brain_globe(data, coor, shift=0, cmap='flag',
                figsize=(12, 12), projection='mollweide'):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    if not isinstance(coor, torch.Tensor):
        coor = torch.tensor(coor)

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
    def __init__(self, data, coor, shift=0,
                 coor_mask=None, struc_tag=None, cmap='flag',
                 figsize=(12, 12), projection='mollweide'):
        self.data, self.coor = self._select_data_and_coor(
            data, coor,
            coor_mask,
            struc_tag
        )
        self.shift = shift
        self.cmap = cmap
        self.figsize = figsize
        self.projection = projection

    def __call__(self):
        return brain_globe(
            data=self.data, coor=self.coor,
            shift=self.shift, cmap=self.cmap,
            figsize=self.figsize, projection=self.projection
        )


class _SurfNoActionMixin:
    def _select_data_and_coor(self, data, coor,
                              coor_mask=None,
                              struc_tag=None):
        return data, coor


class _SurfFromFilesMixin:
    def _select_data_and_coor(self, data, coor,
                              coor_mask=None,
                              struc_tag=None):
        data = nb.load(data)
        coor = nb.load(coor).darrays[0].data
        if coor_mask is not None:
            coor_mask = nb.load(coor_mask)
            coor_mask = coor_mask.darrays[0].data.astype(bool)
            coor = coor[coor_mask]
        coor = sphere_to_latlong(torch.tensor(coor))
        if struc_tag is not None:
            slices = []
            brain_model_axis = data.header.get_axis(1)
            for struc, slc, _ in brain_model_axis.iter_structures():
                if struc in struc_tag:
                    slices.append(slc)
            slices = np.r_[tuple(slices)]
            data = data.get_fdata()[:, slices]
        else:
            data = data.get_fdata()
        return torch.tensor(data), coor


class _CortexLfsLR32KMixin(_SurfFromFilesMixin):
    def _select_data_and_coor(self, data):
        return super()._select_data_and_coor(
            data=data,
            coor=tflow.get(**TFLOW_COOR_QUERY, **TFLOW_COMPARTMENTS['L']),
            coor_mask=tflow.get(**TFLOW_MASK_QUERY, **TFLOW_COMPARTMENTS['L']),
            struc_tag='CIFTI_STRUCTURE_CORTEX_LEFT'
        )


class _CortexRfsLR32KMixin(_SurfFromFilesMixin):
    def _select_data_and_coor(self, data):
        return super()._select_data_and_coor(
            data=data,
            coor=tflow.get(**TFLOW_COOR_QUERY, **TFLOW_COMPARTMENTS['R']),
            coor_mask=tflow.get(**TFLOW_MASK_QUERY, **TFLOW_COMPARTMENTS['R']),
            struc_tag='CIFTI_STRUCTURE_CORTEX_RIGHT'
        )


class GlobeBrain(_GlobeBrain, _SurfNoActionMixin):
    pass


class GlobeCortexL(_GlobeBrain, _SurfNoActionMixin):
    def __init__(self, data, coor, cmap='flag',
                 figsize=(12, 12), projection='mollweide'):
        super().__init__(
            data=data, coor=coor, shift=(3 * np.pi / 4),
            cmap=cmap, figsize=figsize, projection=projection
        )


class GlobeCortexR(_GlobeBrain, _SurfNoActionMixin):
    def __init__(self, data, coor, cmap='flag',
                 figsize=(12, 12), projection='mollweide'):
        super().__init__(
            data=data, coor=coor, shift=(np.pi / 4),
            cmap=cmap, figsize=figsize, projection=projection
        )


class GlobeFromFiles(_GlobeBrain, _SurfFromFilesMixin):
    pass
