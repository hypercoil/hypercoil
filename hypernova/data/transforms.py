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
    def __init__(self, dtype='torch.FloatTensor', nanfill=None, names=None):
        self.dtype = dtype
        self.nanfill = nanfill
        self.all_names = names

    def __call__(self, sample):
        path = sample.path if isinstance(sample, BIDSFile) else sample
        data = self.read(path)
        data = torch.Tensor(data).type(self.dtype)
        if self.nanfill is not None:
            data[torch.isnan(data)] = self.nanfill
        data.names = self.names(data)
        return data

    @abstractmethod
    def read(self, path):
        raise NotImplementedError()

    def names(self, data):
        return self.all_names[:data.dim()]


class ReadNiftiTensor(ReadNeuroTensor):
    def __init__(self, dtype='torch.FloatTensor', nanfill=None):
        names = ('x', 'y', 'z', 't')
        super(ReadNiftiTensor, self).__init__(dtype, nanfill, names)

    def read(self, path):
        return nb.load(path).get_fdata()


class ReadTableTensor(ReadNeuroTensor):
    def __init__(self, dtype='torch.FloatTensor', nanfill=None):
        names = ('var', 't')
        super(ReadTableTensor, self).__init__(dtype, nanfill, names)

    def read(self, path):
        return pd.read_csv(path, sep='\t').values


class ReadNeuroTensorBlock(object):
    def __init__(self, dtype='torch.FloatTensor', nanfill=None):
        self.dtype = dtype
        self.nanfill = nanfill

    def __call__(self, sample, new_names=None):
        if isinstance(sample, list):
            if new_names is not None:
                nn = new_names[1:]
            else:
                nn = new_names
            tensors = extend_and_conform([self(s, nn) for s in sample])
            names = tensors[0].shape
            tensors = torch.stack([t.rename(None) for t in tensors])
            if new_names is not None:
                tensors.names = names + new_names
            return tensors
        return self.transform(sample)


def extend_and_conform(tensor_list):
    sizes = [t.size() for t in tensor_list]
    out_size = torch.amax(torch.Tensor(sizes), 0).int()
    return [extend_to_size(t, out_size) for t in tensor_list]


def extend_to_size(tensor, size):
    out = torch.empty(*out_size) * float('nan')
    names = tensor.names
    tensor = tensor.rename(None)
    out[[slice(s) for s in tensor.size()]] = tensor
    out.names = names
    return out


class ReadNiftiTensorBlock(ReadNeuroTensorBlock):
    def __init__(self, dtype='torch.FloatTensor', nanfill=None):
        super(ReadNiftiTensorBlock, self).__init__(dtype, nanfill)
        self.transform = ReadNiftiTensor(self.dtype, self.nanfill)


class ReadTableTensorBlock(ReadNeuroTensorBlock):
    def __init__(self, dtype='torch.FloatTensor', nanfill=None):
        super(ReadTableTensorBlock, self).__init__(dtype, nanfill)
        self.transform = ReadTableTensor(self.dtype, self.nanfill)


class EncodeOneHot(object):
    def __init__(self, n_levels, dtype='torch.FloatTensor'):
        self.n_levels = n_levels
        self.dtype = dtype
        self.patterns = torch.eye(self.n_levels)

    def __call__(self, sample):
        idx = torch.Tensor(sample).type('torch.LongTensor')
        return self.patterns[idx]
