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
from hypercoil.neuro.const import fsLR


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


def data_from_struc_tag(cifti, struc_tag):
    if struc_tag is not None:
        slices = []
        brain_model_axis = cifti.header.get_axis(1)
        for struc, slc, _ in brain_model_axis.iter_structures():
            if struc in struc_tag:
                if slc.stop is None:
                    stop = cifti.shape[-1]
                    slc = slice(slc.start, stop, slc.step)
                slices.append(slc)
        slices = np.r_[tuple(slices)]
        data = cifti.get_fdata()[:, slices]
    else:
        data = cifti.get_fdata()
    return torch.tensor(data)


class _GlobeBrain:
    def __init__(self, data, coor=None, shift=0,
                 coor_mask=None, struc_tag=None, cmap='flag',
                 figsize=(12, 12), projection='mollweide'):
        self.data, self.coor = self._select_data_and_coor(
            data, coor,
            coor_mask,
            struc_tag
        )
        self.cmap = self._select_cmap(cmap, struc_tag)
        self.shift = shift
        self.figsize = figsize
        self.projection = projection

    def drop_null(self, null=0):
        subset = (self.data != null)
        return self.data[subset], self.coor[subset]

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
        data = data_from_struc_tag(data, struc_tag)
        return data, coor


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


class _CMapNoActionMixin:
    def _select_cmap(self, cmap, struc_tag=None):
        return cmap


class _CMapFromSurfMixin:
    def _compute_linear_map(self, null=0):
        labels = np.unique(self.data).astype(int)
        labels = np.delete(labels, labels==null)
        n_labels = labels.max() + 1
        n_voxels = np.prod(self.data.shape[:3])
        map = np.zeros((n_labels, n_voxels))
        for l in labels:
            map[l, :] = (self.data == l)
        map /= map.sum(1, keepdims=True)
        map[np.isnan(map)] = 0
        return torch.tensor(map)

    def _compute_parcel_colours(self, linear_map, surf_cmap):
        parcel_colours = linear_map @ surf_cmap.T
        parcel_colours = parcel_colours.numpy()
        parcel_colours = np.maximum(parcel_colours, 0)
        parcel_colours = np.minimum(parcel_colours, 1)
        return parcel_colours

    def _align_cmap_to_data(self, dataset, cmap):
        null = 0 # hard coding this now
        present_in_dset = np.unique(dataset).astype(int)
        present_in_dset = np.delete(
            present_in_dset,
            present_in_dset==null
        )
        present_map = np.arange(len(present_in_dset)).astype(int) + 1
        present_band = np.zeros(cmap.colors.shape[0]).astype(int)
        present_band[present_in_dset] = present_map
        # This shift sort of depends on where null is, so right now your
        # null had better be zero
        cmap = cmap.colors[present_in_dset]
        dataset = dataset.int()
        dataset = present_band[dataset]
        return dataset, ListedColormap(cmap)

    def _select_cmap(self, cmap, struc_tag=None):
        surf_cmap = nb.load(cmap)
        surf_cmap = data_from_struc_tag(surf_cmap, struc_tag)
        linear_map = self._compute_linear_map(null=0)
        parcel_colours = self._compute_parcel_colours(linear_map, surf_cmap)
        cmap = ListedColormap(parcel_colours)
        self.data, cmap = self._align_cmap_to_data(self.data, cmap)
        self.data, self.coor = self.drop_null(null=0)
        return cmap


class _ModalGlobeMixin:
    def _config_map_and_cmap(self):
        pass


class _NetworkGlobeMixin:
    def _config_map_and_cmap(self):
        pass


class GlobeBrain(_GlobeBrain, _SurfNoActionMixin, _CMapNoActionMixin):
    pass


class GlobeFromFiles(_GlobeBrain, _SurfFromFilesMixin, _CMapNoActionMixin):
    pass


class CortexLfsLRFromFiles(_GlobeCortexL, _CortexLfsLR32KMixin, _CMapFromSurfMixin):
    pass


class CortexRfsLRFromFiles(_GlobeCortexR, _CortexRfsLR32KMixin, _CMapFromSurfMixin):
    pass
