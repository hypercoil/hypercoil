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
from abc import ABC, abstractmethod
from bids.layout.models import BIDSFile


class IdentityTransform(object):
    def __call__(self, sample):
        return sample


class ReadNeuroTensor(ABC):
    def __init__(self, dtype='torch.FloatTensor', nanfill=0):
        self.dtype = dtype
        self.nanfill = nanfill

    def __call__(self, sample):
        path = sample.path if isinstance(sample, BIDSFile) else sample
        data = self.read(path)
        data = torch.Tensor(data).type(self.dtype)
        data[torch.isnan(data)] = self.nanfill
        return data

    @abstractmethod
    def read(self, path):
        raise NotImplementedError()


class ReadNiftiTensor(ReadNeuroTensor):
    def read(self, path):
        return nb.load(path).get_fdata()


class ReadTableTensor(ReadNeuroTensor):
    def read(self, path):
        return pd.read_csv(path, sep='\t').values


class ReadNeuroTensorBlock(object):
    def __init__(self, dtype='torch.FloatTensor', nanfill=0):
        self.dtype = dtype
        self.nanfill = nanfill

    def __call__(self, sample):
        if isinstance(sample, list):
            return torch.stack([self(s) for s in sample]).squeeze()
        return self.transform(sample)


class ReadNiftiTensorBlock(ReadNeuroTensorBlock):
    def __init__(self, dtype='torch.FloatTensor', nanfill=0):
        super(ReadNiftiTensorBlock, self).__init__(dtype, nanfill)
        self.transform = ReadNiftiTensor(self.dtype, self.nanfill)


class ReadTableTensorBlock(ReadNeuroTensorBlock):
    def __init__(self, dtype='torch.FloatTensor', nanfill=0):
        super(ReadTableTensorBlock, self).__init__(dtype, nanfill)
        self.transform = ReadTableTensor(self.dtype, self.nanfill)
