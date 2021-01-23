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
from scipy.ndimage import gaussian_filter
from ..functional import ScalarIIDNoiseSource


class Atlas(object):
    def __init__(self, path, label_dict=None):
        self.path = path
        self.ref = nb.load(self.path)
        self.image = self.ref.get_fdata()
        self.label_dict = label_dict

    def map(self, sigma=None, noise=None, normalise=True):
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


class DiscreteAtlas(Atlas):
    def __init__(self, path, null=0, label_dict=None, mask=None):
        super(DiscreteAtlas, self).__init__(
            path=path, label_dict=label_dict)
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
    def __init__(self, path, label_dict=None, mask=None, thresh=0):
        if isinstance(path, list) or isinstance(path, tuple):
            self._init_from_paths(path, label_dict)
        else:
            super(ContinuousAtlas, self).__init__(
                path=path, label_dict=label_dict)
        self.labels = list(range(self.image.shape[-1]))
        if isinstance(mask, np.ndarray):
            pass
        elif mask == 'auto':
            mask = ((np.abs(self.image)).sum(-1) > thresh)
        elif mask is None:
            mask = np.ones(self.image.shape[:-1])
        self._set_dims(mask)

    def _init_from_paths(self, path, label_dict):
        self.path = path
        self.ref = [nb.load(p) for p in self.path]
        self.image = np.stack([r.get_fdata() for r in self.ref], -1)
        self.label_dict = label_dict

    def _extract_label(self, label_id, sigma=None):
        slc = [slice(None)] * len(self.image.shape)
        slc[-1] = label_id
        map = self.image[tuple(slc)]
        return self._smooth_and_mask(map, sigma)


def atlas_init_(tensor, atlas, kernel_sigma=None, noise_sigma=None, null=0):
    rg = tensor.requires_grad
    tensor.requires_grad = False
    if noise_sigma is not None:
        distr = torch.distributions.normal.Normal(
            torch.Tensor([0]), torch.Tensor([noise_sigma])
        )
        noise = ScalarIIDNoiseSource(distr=distr)
    else:
        noise = None
    map = atlas.map(sigma=kernel_sigma, noise=noise)
    tensor[:] = torch.Tensor(map)
    tensor.requires_grad = rg
