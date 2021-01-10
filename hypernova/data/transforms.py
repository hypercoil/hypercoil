# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Neuroimaging data transforms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Dataset transformations for packaging neuroimaging data into Pytorch tensors.
"""
import torch
from .functional import (
    to_tensor,
    to_named_tensor,
    nanfill,
    apply_model_specs,
    apply_transform,
    transform_block,
    unzip_blocked_dict,
    consolidate_block,
    read_data_frame,
    read_neuro_image,
    vector_encode
)


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
        return to_tensor(data, dtype=self.dtype)


class ToNamedTensor(ToTensor):
    def __init__(self, dtype='torch.FloatTensor',
                 names=None, truncate='last'):
        self.all_names = names
        self.truncate = truncate
        super(ToNamedTensor, self).__init__(dtype)

    def __call__(self, data):
        names = self._names(data)
        return to_named_tensor(data, dtype=dtype, names=names)

    def _names(self, data):
        check = torch.Tensor(data)
        if self.truncate == 'last':
            return self.all_names[:check.dim()]
        elif self.truncate == 'first':
            return self.all_names[-check.dim():]


class NaNFill(object):
    def __init__(self, fill=None):
        self.fill = fill

    def __call__(self, data):
        return nanfill(data, fill=self.fill)


class ApplyModelSpecs(object):
    def __init__(self, models):
        self.models = models

    def __call__(self, data):
        return apply_model_specs(data, models=self.models)


class ApplyTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, iterable):
        return apply_transform(iterable, transform=self.transform)


class BlockTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, block):
        return transform_block(block, transform=self.transform)


class UnzipTransformedBlock(object):
    def __init__(self, depth=-1):
        self.depth = depth
        self.cur_depth = 0

    def __call__(self, block):
        return unzip_blocked_dict(block)


class ConsolidateBlock(object):
    def __call__(self, block):
        return consolidate_block(block)


class ReadDataFrame(object):
    def __init__(self, sep='\t', **kwargs):
        self.sep = sep
        self.kwargs = kwargs

    def __call__(self, path):
        return read_data_frame(path, sep=self.sep, **self.kwargs)


class ReadNeuroImage(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, path):
        return read_neuro_image(path, **self.kwargs)


class EncodeOneHot(object):
    def __init__(self, n_levels, dtype='torch.FloatTensor'):
        self.n_levels = n_levels
        self.dtype = dtype
        self.patterns = torch.eye(self.n_levels)

    def __call__(self, data):
        return vector_encode(data, encoding=self.patterns, dtype=self.dtype)
