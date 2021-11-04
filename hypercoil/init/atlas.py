# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlas initialisation
~~~~~~~~~~~~~~~~~~~~
Tools for initialising parameters corresponding to brain atlases.
"""
import torch
import numpy as np
import nibabel as nb
from functools import partial
from scipy.ndimage import gaussian_filter
from .base import DomainInitialiser
from ..functional import UnstructuredNoiseSource
from ..functional.domain import Identity
from ..functional.sphere import spherical_conv, euclidean_conv


class _SingleReferenceMixin:
    def _load_reference(self, path):
        ref = nb.load(path)
        self.imshape = ref.get_fdata().shape[:3]
        return ref


class _MultiReferenceMixin:
    def _load_reference(self, paths):
        ref = [nb.load(path) for path in paths]
        self.imshape = ref[0].get_fdata().shape[:3]
        return ref


class _TelescopeCfgMixin:
    def _configure_maps(self, X):
        maps = np.zeros((self.n_labels, *self.imshape))
        X = X.get_fdata().squeeze()
        for i, l in enumerate(self.labels):
            maps[i] = (X == l)
        return maps.squeeze()

    def _configure_labels(self, null):
        self.labels = set(np.unique(self.ref.get_fdata())) - set([null])
        self.labels = list(self.labels)
        self.labels.sort()


class _ConcatenateCfgMixin:
    def _configure_maps(self, X):
        maps = np.zeros((self.n_labels, *self.imshape))
        for i, l in enumerate(X):
            maps[i] = l.get_fdata()
        return maps.squeeze()

    def _configure_labels(self, null=None):
        self.labels = list(range(len(self.ref)))



class _AxisRollCfgMixin:
    def _configure_maps(self, X):
        return np.moveaxis(X.get_fdata(), (0, 1, 2, 3), (3, 0, 1, 2)).squeeze()

    def _configure_labels(self, null=None):
        self.labels = list(range(self.ref.shape[-1]))



class _EvenlySampledConvMixin:
    def _configure_sigma(self, sigma):
        if sigma is not None:
            scale = self.ref.header.get_zooms()[:3]
            sigma = [sigma / s for s in scale]
            sigma = [0] + [sigma]
        return sigma

    def _convolve(self, sigma, map):
        if sigma is not None:
            gaussian_filter(map, sigma=sigma, output=map)
        return map


class _SpatialConvMixin:
    def _configure_sigma(self, sigma):
        return sigma

    def _convolve(self, sigma, map):
        if sigma is not None:
            coors = torch.tensor(self.coors)
            map = torch.tensor(map)
            for compartment, slc in self.compartments.items():
                if compartment =='subcortex':
                    map[:, slc] = euclidean_conv(
                        data=map[:, slc].T, coor=coors[slc],
                        scale=sigma, max_bin=self.max_bin,
                        truncate=self.truncate
                    ).T
                else:
                    map[:, slc] = spherical_conv(
                        data=map[:, slc].T, coor=coors[slc],
                        scale=(self.spherical_scale * sigma), r=100,
                        max_bin=self.max_bin, truncate=self.truncate
                    ).T
        return map


class Atlas:
    def __init__(self, path, name=None, mask=None,
                 label_dict=None, thresh=0, null=0):
        self.path = path
        self.ref = self._load_reference(path)
        self.name = name or 'atlas'
        self.label_dict = label_dict
        self._configure_labels(null)
        self.n_labels = len(self.labels)
        map = self._configure_maps(self.ref)
        self.mask = self._configure_mask(mask, thresh, map)
        self.n_voxels = self.mask.sum()

    def map(self, sigma=None, noise=None, normalise=True):
        map = self._configure_maps(self.ref)
        sigma = self._configure_sigma(sigma)
        map = self._convolve(sigma, map)
        map = torch.tensor(self.unfold(map))
        if noise is not None:
            map = noise(map)
        if normalise:
            map /= map.sum(1, keepdim=True)
        return map

    def unfold(self, map):
        return map[:, self.mask].reshape(self.n_labels, self.n_voxels)

    def fold(self, map):
        folded_map = np.zeros((self.n_labels, *self.imshape))
        folded_map[:, self.mask] = map
        return folded_map

    def foldmax(self, map):
        map = self.fold(map)
        return map.max(0)

    def _set_dims(self, mask):
        self.mask = mask

    def _configure_mask(self, mask, thresh, map):
        if isinstance(mask, np.ndarray):
            pass
        elif mask == 'auto':
            mask = (map.sum(0) > thresh)
        elif mask is None:
            mask = np.ones(self.imshape)
        return mask.squeeze().astype(bool)


class AtlasWithCoordinates(Atlas):
    def __init__(self, path, surf_L=None, surf_R=None, name=None,
                 label_dict=None, mask_L=None, mask_R=None, null=0,
                 cortex_L='CIFTI_STRUCTURE_CORTEX_LEFT',
                 cortex_R='CIFTI_STRUCTURE_CORTEX_RIGHT',
                 max_bin=10000, truncate=None, spherical_scale=1.):
        super(AtlasWithCoordinates, self).__init__(
            path, name=name, label_dict=label_dict, mask=None, null=null
        )
        self.surf = {
            'L': surf_L,
            'R': surf_R
        }
        self.masks = {
            'L': mask_L,
            'R': mask_R
        }
        self.cortex = {
            'L': cortex_L,
            'R': cortex_R
        }
        self.max_bin = max_bin
        self.truncate = truncate
        self.spherical_scale = spherical_scale
        self._init_coors()

    def dump_coors(self, compartments=None):
        if compartments is None:
            return self.coors
        compartments = np.r_[
            tuple([self.compartments[c] for c in compartments])
        ]
        return self.coors[compartments]

    def _init_coors(self):
        self._compartment_masks()
        _, model_axis = self._cifti_model_axes()
        cortex = {
            'L': self.compartments['cortex_L'],
            'R': self.compartments['cortex_R'],
        }
        sub = self.compartments['subcortex']
        self.coors = np.zeros((self.n_voxels, 3))
        self.coors[sub] = model_axis.voxel[sub]
        for hemi in ('L', 'R'):
            if self.surf[hemi] is None:
                continue
            coor = nb.load(self.surf[hemi])
            if self.masks[hemi] is not None:
                mask = nb.load(self.masks[hemi])
                mask = mask.darrays[0].data.astype(bool)
                coor = coor.darrays[0].data[mask]
            else:
                coor = coor.darrays[0].data
            self.coors[cortex[hemi]] = coor

    def _compartment_masks(self):
        self.compartments = {
            'cortex_L' : slice(0, 0),
            'cortex_R' : slice(0, 0),
            'subcortex': slice(0, 0),
        }
        _, model_axis = self._cifti_model_axes()
        for struc, slc, _ in (model_axis.iter_structures()):
            if struc == self.cortex['L']:
                self.compartments['cortex_L'] = slc
            elif struc == self.cortex['R']:
                self.compartments['cortex_R'] = slc
        try:
            vol_mask = np.where(model_axis.volume_mask)[0]
            vol_min, vol_max = vol_mask.min(), vol_mask.max() + 1
            self.compartments['subcortex'] = slice(vol_min, vol_max)
        except ValueError:
            pass

    def _cifti_model_axes(self):
        """
        Thanks to Chris Markiewicz for tutorials that shaped this
        implementation.
        """
        return [
            self.ref.header.get_axis(i)
            for i in range(self.ref.ndim)
        ]


class DiscreteAtlas(
    Atlas,
    _SingleReferenceMixin,
    _TelescopeCfgMixin,
    _EvenlySampledConvMixin
):
    pass


class MultivolumeAtlas(
    Atlas,
    _SingleReferenceMixin,
    _AxisRollCfgMixin,
    _EvenlySampledConvMixin
):
    pass


class MultifileAtlas(
    Atlas,
    _MultiReferenceMixin,
    _ConcatenateCfgMixin,
    _EvenlySampledConvMixin
):
    pass


class SurfaceAtlas(
    AtlasWithCoordinates,
    _SingleReferenceMixin,
    _TelescopeCfgMixin,
    _SpatialConvMixin
):
    pass


def atlas_init_(tensor, atlas, kernel_sigma=None, noise_sigma=None,
                normalise=False):
    """
    Voxel-to-label mapping initialisation.

    Initialise a tensor such that its entries characterise a matrix that maps
    a relevant subset of image voxels to a set of labels. The initialisation
    uses an existing atlas with the option of blurring labels or injecting
    noise.

    Dimension
    ---------
    - tensor : :math:`(L, V)`
      L denotes the total number of labels in the atlas, and V denotes the
      number of voxels to be labelled.

    Parameters
    ----------
    tensor : Tensor
        Tensor to initialise in-place.
    atlas : Atlas object
        Atlas object to use for tensor initialisation.
    kernel_sigma : float or None (default None)
        If this is a float, then a Gaussian smoothing kernel with the
        specified width is applied to each label after it is extracted.
    noise_sigma : float or None (default None)
        If this is a float, then Gaussian noise with the specified standard
        deviation is added to the label.
    """
    if noise_sigma is not None:
        distr = torch.distributions.normal.Normal(
            torch.Tensor([0]), torch.Tensor([noise_sigma])
        )
        noise = UnstructuredNoiseSource(distr=distr)
    else:
        noise = None
    val = atlas.map(sigma=kernel_sigma,
                    noise=noise,
                    normalise=normalise)
    tensor.copy_(val)


class AtlasInit(DomainInitialiser):
    def __init__(self, atlas, kernel_sigma=None, noise_sigma=None,
                 normalise=False, domain=None):
        init = partial(atlas_init_, atlas=atlas, kernel_sigma=kernel_sigma,
                       noise_sigma=noise_sigma, normalise=normalise)
        super(AtlasInit, self).__init__(init=init, domain=domain)
