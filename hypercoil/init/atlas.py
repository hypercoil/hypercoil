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


class Atlas(object):
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
    def __init__(self, path, name=None, label_dict=None):
        self.path = path
        self.ref = nb.load(self.path)
        self.image = self.ref.get_fdata()
        self.name = name or 'atlas'
        self.label_dict = label_dict

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
        map = np.zeros((self.n_labels, self.n_voxels))
        for i, l in enumerate(self.labels):
            map[i] = self._extract_label(l, sigma)
        map = torch.Tensor(map)
        if noise is not None:
            map = noise(map)
        if normalise:
            map /= map.sum(1, keepdim=True)
        return map

    def _set_dims(self, mask):
        self.mask = mask.astype(np.bool)
        self.n_labels = len(self.labels)
        self.n_voxels = self.mask.sum()

    def _smooth_and_mask(self, map, sigma):
        if sigma is not None:
            gaussian_filter(map, sigma=sigma, output=map)
        return map[self.mask]

    def __repr__(self):
        s = f'{type(self).__name__}({self.name}, '
        s += f'labels={self.n_labels}, voxels={self.n_voxels}'
        s += ')'
        return s


class DiscreteAtlas(Atlas):
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
    def __init__(self, path, name=None, label_dict=None, mask=None, null=0):
        super(DiscreteAtlas, self).__init__(
            path=path, name=name, label_dict=label_dict)
        self.labels = set(np.unique(self.image)) - set([null])
        if isinstance(mask, np.ndarray):
            pass
        elif mask == 'auto':
            mask = (self.image != null)
        elif mask is None:
            mask = np.ones_like(self.image)
        self._set_dims(mask)

    def _extract_label(self, label_id, sigma=None):
        map = (self.image == label_id).astype(np.float)
        return self._smooth_and_mask(map, sigma)


class ContinuousAtlas(Atlas):
    """
    Continuous atlas container object. Use for atlases whose labels overlap and
    must therefore be stored across multiple image volumes -- for instance,
    probabilistic segmentations or ICA results.

    Parameters
    ----------
    path : str or pathlib.Path or list
        Path to a NIfTI file containing the atlas. If this is a list, then each
        entry in the list will be interpreted as a separate atlas label.
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
    def __init__(self, path, name=None, label_dict=None, mask=None, thresh=0):
        if isinstance(path, list) or isinstance(path, tuple):
            self._init_from_paths(path=path, name=name, label_dict=label_dict)
        else:
            super(ContinuousAtlas, self).__init__(
                path=path, name=name, label_dict=label_dict)
        self.labels = list(range(self.image.shape[-1]))
        if isinstance(mask, np.ndarray):
            pass
        elif mask == 'auto':
            mask = ((np.abs(self.image)).sum(-1) > thresh)
        elif mask is None:
            mask = np.ones(self.image.shape[:-1])
        self._set_dims(mask)

    def _init_from_paths(self, path, name, label_dict):
        self.path = path
        self.ref = [nb.load(p) for p in self.path]
        self.image = np.stack([r.get_fdata() for r in self.ref], -1)
        self.label_dict = label_dict
        self.name = name or 'atlas'

    def _extract_label(self, label_id, sigma=None):
        slc = [slice(None)] * len(self.image.shape)
        slc[-1] = label_id
        map = self.image[tuple(slc)]
        return self._smooth_and_mask(map, sigma)


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
