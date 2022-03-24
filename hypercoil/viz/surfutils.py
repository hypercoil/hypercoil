# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Brain surface plotting utilities.
"""
import torch
import numpy as np
import nibabel as nb
from matplotlib.colors import ListedColormap


def data_from_struc_tag(cifti, struc_tag, dtype=None, device=None):
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
    return torch.tensor(data, dtype=dtype, device=device)


class _SurfNoActionMixin:
    def _select_data_and_coor(self, data, coor,
                              coor_mask=None,
                              struc_tag=None):
        return data, coor


class _SurfFromFilesMixin:
    def _select_data_and_coor(self, data, coor,
                              coor_mask=None,
                              struc_tag=None,
                              dtype=None, device=None):
        data = nb.load(data)
        coor = nb.load(coor).darrays[0].data
        if coor_mask is not None:
            coor_mask = nb.load(coor_mask)
            coor_mask = coor_mask.darrays[0].data.astype(bool)
            coor = coor[coor_mask]
        data = data_from_struc_tag(data, struc_tag)
        return data, coor


class _CMapNoActionMixin:
    def _select_cmap(self, cmap, struc_tag=None):
        return cmap


class _CMapFromSurfMixin:
    def _compute_linear_map(self, null=0, dtype=None, device=None):
        labels = np.unique(self.data).astype(int)
        labels = np.delete(labels, labels==null)
        n_labels = labels.max() + 1
        n_voxels = np.prod(self.data.shape[:3])
        map = np.zeros((n_labels, n_voxels))
        for l in labels:
            map[l, :] = (self.data == l)
        map /= map.sum(1, keepdims=True)
        map[np.isnan(map)] = 0
        return torch.tensor(map, dtype=dtype, device=device)

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
