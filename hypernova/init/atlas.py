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


class DiscreteAtlas(object):
    def __init__(self, path, null=0, label_dict=None, mask=None):
        self.path = path
        self.ref = nb.load(self.path)
        self.image = self.ref.get_fdata()
        self.labels = set(np.unique(self.image)) - set([null])
        if mask == 'auto':
            mask = (self.image != null)
        elif mask is None:
            mask = np.ones_like(self.image)
        self.mask = mask.astype(np.bool)
        self.n_labels = len(self.labels)
        self.n_voxels = self.mask.sum()

    def _extract_label(self, label_id, sigma=None):
        map = (self.image == label_id).astype(np.float)
        if sigma is not None:
            gaussian_filter(map, sigma=sigma, output=map)
        return map[self.mask]

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
