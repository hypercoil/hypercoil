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
from ..formula import ModelSpec
from .grabber import LightBIDSObject


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class IdentityTransform(object):
    def __call__(self, data):
        return data


class ToTensor(object):
    def __init__(self, dtype='torch.FloatTensor'):
        self.dtype = dtype

    def __call__(self, data):
        if isinstance(data, pd.DataFrame):
            data = data.values
        return torch.Tensor(data).type(self.dtype)


class ToNamedTensor(ToTensor):
    def __init__(self, dtype='torch.FloatTensor',
                 names=None, truncate='last'):
        self.all_names = names
        self.truncate = truncate
        super(ToNamedTensor, self).__init__(dtype)

    def __call__(self, data):
        out = super(ToNamedTensor, self).__call__(data)
        out.names = self.names(out)
        return out

    def names(self, data):
        if self.truncate == 'last':
            return self.all_names[:data.dim()]
        elif self.truncate == 'first':
            return self.all_names[-data.dim():]


class NaNFill(object):
    def __init__(self, nanfill=None):
        self.nanfill = nanfill

    def __call__(self, data):
        if self.nanfill is not None:
            if isinstance(data. np.ndarray):
                data[np.isnan(data)] = self.nanfill
            elif isinstance(data, torch.Tensor):
                data[torch.isnan(data)] = self.nanfill
        return data


class ApplyModelSpecs(object):
    def __init__(self, models):
        self.models = models

    def __call__(self, data):
        return {m.name: m(data) for m in self.models}


class ApplyTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, iterable):
        if isinstance(iterable, dict):
            return {k: self.transform(v) for k, v in iterable.items()}
        else:
            return [self.transform(v) for v in iterable]


class BlockTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, block):
        if isinstance(block, list):
            return [self(e) for e in block]
        return self.transform(block)


class UnzipTransformedBlock(object):
    def __init__(self, depth=-1):
        self.depth = depth
        self.cur_depth = 0

    def __call__(self, block):
        if isinstance(block[0], list):
            return self([self(e) for e in block])
        return {k: [d[k] for d in block] for k in block[0].keys()}


class ConsolidateBlock(object):
    def __call__(self, block):
        if isinstance(block, torch.Tensor):
            return block
        elif isinstance(block[0], torch.Tensor):
            out = extend_and_conform(block)
        else:
            out = extend_and_conform([self(e) for e in block])
        return torch.stack(out)


def extend_and_conform(tensor_list):
    sizes = [t.size() for t in tensor_list]
    out_size = torch.amax(torch.Tensor(sizes), 0).int()
    return [extend_to_size(t, out_size) for t in tensor_list]


def extend_to_size(tensor, size):
    out = torch.empty(*size) * float('nan')
    names = tensor.names
    tensor = tensor.rename(None)
    out[[slice(s) for s in tensor.size()]] = tensor
    out.names = names
    return out


class ReadDataFrame(object):
    def __init__(self, sep='\t', **kwargs):
        self.sep = sep
        self.kwargs = kwargs

    def __call__(self, path):
        if isinstance(path, LightBIDSObject):
            path = path.path
        return pd.read_csv(path, sep=self.sep, **self.kwargs)


class ReadNeuroImage(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, path):
        if isinstance(path, LightBIDSObject):
            path = path.path
        return nb.load(path, **self.kwargs).get_fdata()
