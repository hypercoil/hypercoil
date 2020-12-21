# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Neuroimaging data transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dataset transformations for packaging neuroimaging data into Pytorch tensors.
"""
import torch
import nibabel as nb
import pandas as pd
from bids.layout.models import BIDSFile


class ReadNiftiTensor(object):
    def __init__(self, dtype='torch.FloatTensor', nanfill=0):
        self.dtype = dtype
        self.nanfill = nanfill

    def __call__(self, sample):
        path = sample.path if isinstance(sample, BIDSFile) else sample
        img = nb.load(path)
        img = torch.Tensor(img.get_fdata()).type(self.dtype)
        img[torch.isnan(img)] = self.nanfill
        return img


class ReadTableTensor(object):
    def __init__(self, dtype='torch.FloatTensor', nanfill=0):
        self.dtype = dtype
        self.nanfill = nanfill

    def __call__(self, sample):
        path = sample.path if isinstance(sample, BIDSFile) else sample
        data = pd.read_csv(path, sep='\t')
        data = torch.Tensor(data.values).type(self.dtype)
        data[torch.isnan(data)] = self.nanfill
        return data


class IdentityTransform(object):
    def __call__(self, sample):
        return sample
