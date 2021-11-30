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


#TODO: restore doc strings when github stops being absolute garbage


class _SingleReferenceMixin:
    def _load_reference(self, path):
        ref = nb.load(path)
        self.imshape = ref.get_fdata().shape[:3]
        return ref


class _MultiReferenceMixin:
    def _load_reference(self, paths):
        ref = [nb.load(path) for path in paths]
        self.imshape = ref[0].get_fdata().shape[:3]
        #TODO: eventually we might wish to drop _ConcatenateCfgMixin. This
        # could give us more interoperability between single- and multi-
        # compartment functionality so that these are proper mixins
        #ref = nb.Nifti1Image(
        #    dataobj=np.stack([r.get_fdata() for r in ref]),
        #    affine=ref[0].affine,
        #    header=ref[0].header
        #)
        return ref


class _TelescopeCfgMixin:
    def _configure_maps(self, X, labels, space_dims):
        n_labels = len(labels)
        maps = np.zeros((n_labels, *space_dims))
        for i, l in enumerate(labels):
            maps[i] = (X == l)
        return maps.squeeze()

    def _configure_labels(self, data, null):
        labels = set(np.unique(data)) - set([null])
        labels = list(labels)
        labels.sort()
        return labels


class _ConcatenateCfgMixin:
    def _configure_maps(self, X, labels, space_dims):
        n_labels = len(labels)
        maps = np.zeros((n_labels, *space_dims))
        for i, l in enumerate(X):
            maps[i] = l.get_fdata()
        return maps.squeeze()

    def _configure_labels(self, data, null=None):
        return list(range(len(data)))



class _AxisRollCfgMixin:
    def _configure_maps(self, X, labels, space_dims):
        return np.moveaxis(X.get_fdata(), (0, 1, 2, 3), (3, 0, 1, 2)).squeeze()

    def _configure_labels(self, data, null=None):
        return list(range(data.shape[-1]))


class _SingleCompartmentMixin:
    def unfold(self, map):
        return map[:, self.mask].reshape(self.n_labels, self.n_voxels)

    def fold(self, map):
        folded_map = np.zeros((self.n_labels, *self.imshape))
        folded_map[:, self.mask] = map
        return folded_map

    def foldmax(self, map):
        map = self.fold(map)
        return map.max(0)

    def _compartment_masks(self):
        self.compartments = None #{'all' : slice(None, None, None)}

    def _configure_all_labels(self, null):
        try:
            data = self.ref.get_fdata()
        except AttributeError:
            data = self.ref
        self.labels = self._configure_labels(data, null)
        self.n_labels = len(self.labels)

    def _configure_all_maps(self, img):
        try:
            X = img.get_fdata().squeeze()
        except AttributeError:
            X = img
        return self._configure_maps(X, self.labels, self.imshape)

    def _normalise_all_maps(self, map):
        return map / map.sum(1, keepdim=True)

    def _to_tensor(self, map):
        return torch.tensor(map)


class _MultiCompartmentMixin:
    def unfold(self, map):
        return map

    def fold(self, map):
        return map # only greyordinates as of now
        # TODO: handle these reasonably
        #for c, m in map.items():
        #    map

    def foldmax(self, map):
        map = self.fold(map)
        for c, m in map.items():
            map[c] = map[c].max(0)
        #TODO: plug it into a single array using the compartment indices
        return map

    def _compartment_masks(self):
        self.compartments = {'all' : slice(None, None, None)}

    def _configure_all_labels(self, null):
        self.labels = {}
        self.n_labels = {}
        for c, slc in self.compartments.items():
            #TODO: this is not going to work as expected when the reference is
            # derived from a list of files
            data = self.ref.get_fdata().squeeze()[slc]
            self.labels[c] = self._configure_labels(data, null)
            self.n_labels[c] = len(self.labels[c])

    def _configure_all_maps(self, img):
        maps = {}
        for c, slc in self.compartments.items():
            X = img.get_fdata().squeeze()[slc]
            maps[c] = self._configure_maps(
                X, self.labels[c], [slc.stop - slc.start]
            )
        return maps

    def _normalise_all_maps(self, map):
        map_n = {}
        for c, m in map.items():
            map_n[c] = map[c] / map[c].sum(1, keepdim=True)
        return map_n

    def _to_tensor(self, map):
        map_t = {}
        for c, m in map.items():
            map_t[c] = torch.tensor(m)
        return map_t


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
    """
    Atlas object for linear mapping from voxels to labels.

    Base class inherited by discrete and continuous atlas containers.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a NIfTI file containing the atlas.
    label_dict : dict or None (default None)
        Dictionary mapping labels or volumes in the image to parcel or region
        names or identifiers.

    Attributes
    ----------
    ref : nb.Nifti1Image
        NIfTI image object container for the atlas data.
    image : np.ndarray
        Atlas volume(s).
    mask : np.ndarray
        Mask indicating the voxels to include in the mapping.
    labels : set
        Unique labels in the atlas.
    n_labels : int
        Total number of labels, parcels, regions, or volumes in the atlas.
    n_voxels : int
        Total number of voxels to include in the atlas.
    """
    def __init__(self, path, name=None, mask=None,
                 label_dict=None, thresh=0, null=0):
        self.path = path
        self.ref = self._load_reference(path)
        self.name = name or 'atlas'
        self.label_dict = label_dict
        self._compartment_masks()
        self._configure_all_labels(null)
        map = self._configure_all_maps(self.ref)
        self.mask = self._configure_mask(mask, thresh, map)
        self.n_voxels = self.mask.sum()

    def map(self, sigma=None, noise=None, normalise=True):
        """
        Obtain a matrix representation of the linear mapping from mask voxels
        to atlas labels.

        Parameters
        ----------
        sigma : float or None (default None)
            If this is a float, then a Gaussian smoothing kernel with the
            specified width is applied to each label after it is extracted.
        noise : UnstructuredNoiseSource object or None (default None)
            If this is a noise source, then noise sampled from the source is
            added to the label.
        normalise : bool (default True)
            Indicates whether the result should be normalised such that each
            label time series is a weighted mean over voxel time series.

        Returns
        -------
        map : Tensor
            A matrix representation of the linear mapping from mask voxels to
            atlas labels.

        The order of operations is:
        1. Label extraction
        2. Gaussian spatial filtering
        3. Casting to Tensor
        4. IID noise injection
        5. Normalisation
        """
        map = self._configure_all_maps(self.ref)
        sigma = self._configure_sigma(sigma)
        map = self._convolve(sigma, map)
        map = self._to_tensor(self.unfold(map))
        if noise is not None:
            map = noise(map)
        if normalise:
            return self._normalise_all_maps(map)
        return map

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


class _AtlasWithCoordinates(Atlas, _MultiCompartmentMixin):
    def __init__(self, path, surf_L=None, surf_R=None, name=None,
                 label_dict=None, mask_L=None, mask_R=None, null=0,
                 cortex_L='CIFTI_STRUCTURE_CORTEX_LEFT',
                 cortex_R='CIFTI_STRUCTURE_CORTEX_RIGHT',
                 max_bin=10000, truncate=None, spherical_scale=1.):
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
        super(_AtlasWithCoordinates, self).__init__(
            path, name=name, label_dict=label_dict, mask=None, null=null
        )
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
            if slc.stop is None:
                slc = slice(slc.start, model_axis.size, slc.step)
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
    _SingleCompartmentMixin,
    _EvenlySampledConvMixin
):
    """
    Discrete atlas container object. Use for atlases stored in single-volume
    images with non-overlapping, discrete-valued parcels.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a NIfTI file containing the atlas.
    label_dict : dict or None (default None)
        Dictionary mapping labels in the image to parcel or region names or
        identifiers.
    mask : np.ndarray or 'auto' or None (default None)
        Mask indicating the voxels to include in the mapping. If this is
        'auto', then a mask is automatically formed from voxels with non-null
        values (before smoothing).
    null : float (default 0)
        Value in the image indicating that the voxel belongs to no label. To
        assign every voxel a label, specify a number not in the image.

    Attributes
    ----------
    ref : nb.Nifti1Image
        NIfTI image object container for the atlas data.
    image : np.ndarray
        Atlas volume.
    labels : set
        Unique labels in the atlas.
    n_labels : int
        Total number of labels, parcels, or regions in the atlas.
    n_voxels : int
        Total number of voxels to include in the atlas.
    """


class MultivolumeAtlas(
    Atlas,
    _SingleReferenceMixin,
    _AxisRollCfgMixin,
    _SingleCompartmentMixin,
    _EvenlySampledConvMixin
):
    """
    Continuous atlas container object. Use for atlases whose labels overlap and
    must therefore be stored across multiple image volumes -- for instance,
    probabilistic segmentations or ICA results.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a NIfTI file containing the atlas.
    label_dict : dict or None (default None)
        Dictionary mapping labels or volumes in the image to parcel or region
        names or identifiers.
    mask : np.ndarray or 'auto' or None (default None)
        Mask indicating the voxels to include in the mapping. If this is
        'auto', then a mask is automatically formed from voxels with non-null
        values (before smoothing).
    thresh : float (default 0)
        Threshold for auto-masking.

    Attributes
    ----------
    ref : nb.Nifti1Image
        NIfTI image object container for the atlas data.
    image : np.ndarray
        Atlas volume.
    mask : np.ndarray
        Mask indicating the voxels to include in the mapping.
    labels : set
        Unique labels in the atlas.
    n_labels : int
        Total number of labels, parcels, regions, or volumes in the atlas.
    n_voxels : int
        Total number of voxels to include in the atlas.
    """


class MultifileAtlas(
    Atlas,
    _MultiReferenceMixin,
    _ConcatenateCfgMixin,
    _SingleCompartmentMixin,
    _EvenlySampledConvMixin
):
    """
    Continuous atlas container object. Use for atlases whose labels overlap and
    must therefore be stored across multiple image files -- for instance,
    probabilistic segmentations or ICA results.

    Parameters
    ----------
    path : list(str) or list(pathlib.Path)
        List of paths to NIfTI files containing the atlas. Each entry in the
        list will be interpreted as a separate atlas label.
    label_dict : dict or None (default None)
        Dictionary mapping labels or volumes in the image to parcel or region
        names or identifiers.
    mask : np.ndarray or 'auto' or None (default None)
        Mask indicating the voxels to include in the mapping. If this is
        'auto', then a mask is automatically formed from voxels with non-null
        values (before smoothing).
    thresh : float (default 0)
        Threshold for auto-masking.

    Attributes
    ----------
    ref : nb.Nifti1Image
        NIfTI image object container for the atlas data.
    image : np.ndarray
        Atlas volume.
    mask : np.ndarray
        Mask indicating the voxels to include in the mapping.
    labels : set
        Unique labels in the atlas.
    n_labels : int
        Total number of labels, parcels, regions, or volumes in the atlas.
    n_voxels : int
        Total number of voxels to include in the atlas.
    """


class SurfaceAtlas(
    _AtlasWithCoordinates,
    _SingleReferenceMixin,
    _TelescopeCfgMixin,
    _SpatialConvMixin
):
    """
    CIfTI surface-based atlas container object.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a CIfTI file containing the atlas.
    surf_L and surf_R : str or pathlib.Path
        Paths to GIfTI files containing coordinates of the cortical surfaces of
        the left and right hemispheres. Only spherical coordinates are
        currently supported.
    label_dict : dict or None (default None)
        Dictionary mapping labels or volumes in the image to parcel or region
        names or identifiers.
    mask_L and mask_R : str or pathlib.Path (default None)
        GIfTI files containing masks that are immediately to be applied to
        `surf_L` and `surf_R`. These can be used to exclude coordinates not
        annotated by the CIfTI parcellation -- for instance, the medial wall.
    cortex_L and cortex_R : str
        Names of brain model axis objects corresponding to cortex. Default to
        'CIFTI_STRUCTURE_CORTEX_LEFT' and 'CIFTI_STRUCTURE_CORTEX_RIGHT'.
    max_bin : int (default 10000)
        Spatial convolution parameter. Maximum number of voxels considered per
        convolution. If you run out of memory, try decreasing this.
    truncate : float (default None)
        Spatial convolution parameter. Maximum kernel radius for convolution:
        all data outside of this distance from a given point will not be
        convolved into that point.
    spherical_scale : float (default 1)
        Spatial convolution parameter. Allows setting different sigmas for the
        convolutions on volumetric and spherical data. The sigma for volumetric
        data will be as provided to the `map` function, and the sigma for
        spherical data will be the provided sigma multiplied by the scaling
        factor `spherical_scale`.
    """


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
