# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data transforms
~~~~~~~~~~~~~~~
Dataset transformations for packaging data into Pytorch tensors.
"""
import torch
from . import functional as F


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


class DataObjectTransform(object):
    def __call__(self, data):
        data.update(
            data=super().__call__(data.data)
        )
        return data


class DataObjectTransformWithMetadata(object):
    def __call__(self, data):
        data.update(
            data=super().__call__(data.data, metadata=data.metadata)
        )
        return data


class ToTensor(object):
    def __init__(self, dtype='torch.FloatTensor'):
        self.dtype = dtype

    def __call__(self, data):
        return F.to_tensor(data, dtype=self.dtype)


class ToNamedTensor(ToTensor):
    def __init__(self, dtype='torch.FloatTensor',
                 names=None, truncate='last'):
        self.all_names = names
        self.truncate = truncate
        super(ToNamedTensor, self).__init__(dtype)

    def __call__(self, data):
        names = self._names(data)
        return F.to_named_tensor(data, dtype=dtype, names=names)

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
        return F.nanfill(data, fill=self.fill)


class ApplyModelSpecs(object):
    def __init__(self, models):
        self.models = models

    def __call__(self, data, metadata=None):
        return F.apply_model_specs(
            models=self.models, data=data, metadata=metadata)


class ApplyTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, iterable):
        return F.apply_transform(iterable, transform=self.transform)


class BlockTransform(object):
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, block):
        return F.transform_block(block, transform=self.transform)


class UnzipTransformedBlock(object):
    def __init__(self, depth=-1):
        self.depth = depth
        self.cur_depth = 0

    def __call__(self, block):
        return F.unzip_blocked_dict(block)


class ConsolidateBlock(object):
    def __call__(self, block):
        return F.consolidate_block(block)


class ReadDataFrame(object):
    def __init__(self, sep='\t', **kwargs):
        self.sep = sep
        self.kwargs = kwargs

    def __call__(self, path):
        return F.read_data_frame(path, sep=self.sep, **self.kwargs)


class ReadNeuroImage(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, path):
        return F.read_neuro_image(path, **self.kwargs)


class EncodeOneHot(object):
    def __init__(self, n_levels, dtype='torch.FloatTensor'):
        self.n_levels = n_levels
        self.dtype = dtype
        self.patterns = torch.eye(self.n_levels)

    def __call__(self, data):
        return F.vector_encode(data, encoding=self.patterns, dtype=self.dtype)


class ReadJSON(object):
    def __call__(self, path):
        return F.read_json(path)


class ChangeExtension(object):
    def __init__(self, new_ext, mode='all'):
        self.new_ext = new_ext
        self.mode = mode

    def __call__(self, path):
        return F.change_extension(path, new_ext=self.new_ext, mode=self.mode)


class ToTensorX(DataObjectTransform, ToTensor):
    pass


class ToNamedTensorX(DataObjectTransform, ToNamedTensor):
    pass


class NaNFillX(DataObjectTransform, NaNFill):
    pass


class ApplyModelSpecsX(DataObjectTransformWithMetadata, ApplyModelSpecs):
    pass


class ReadDataFrameX(DataObjectTransform, ReadDataFrame):
    pass


class ReadNeuroImageX(DataObjectTransform, ReadNeuroImage):
    pass


class ReadJSONX(DataObjectTransform, ReadJSON):
    pass


class ChangeExtensionX(DataObjectTransform, ChangeExtension):
    pass
