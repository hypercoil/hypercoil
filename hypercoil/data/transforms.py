# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data transforms
~~~~~~~~~~~~~~~
Dataset transformations for loading data and packaging into Pytorch tensors.

Transformations that operate on DataObjectVariables are denoted with final ~X.
"""
import torch
from textwrap import indent
from . import functional as F


INDENT = '    '


class Compose(object):
    """
    Compose a sequence of transforms. Identical to torchvision's Compose.

    As `Compose` does not subclass Module and as it readily operates on
    non-Tensor data types, it is incompatible with torchscript. Composition
    of compatible transforms can be implemented using `torch.nn.Sequential`.

    Parameters
    ----------
    transforms : list
        List of transforms to compose. Each transform should be a callable
        that accepts as input the expected output of the previous transform
        in the chain.
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        s = f'{type(self).__name__}('
        for t in self.transforms:
            try:
                s += indent(f'\n{t}', INDENT)
            except TypeError:
                s += indent(f'\n{type(t).__name__}()', INDENT)
        s += '\n)'
        return s


class IdentityTransform(object):
    """
    A transform that does nothing. Included to support uniform interfaces.
    """
    def __call__(self, data):
        return data

    def __repr__(self):
        return f'{type(self).__name__}()'


class DataObjectTransform(object):
    """
    When inherited first, sets a transform to operate on the `data` field of
    a DataObjectVariable assignment.
    """
    def __call__(self, data):
        data.update(
            data=super().__call__(data.data)
        )
        return data


class DataObjectTransformWithMetadata(object):
    """
    When inherited first, sets a transform to operate on the `data` field of
    a DataObjectVariable assignment while enabling use of information from the
    `metadata` field.
    """
    def __call__(self, data):
        data.update(
            data=super().__call__(data.data, metadata=data.metadata)
        )
        return data


class ToTensor(object):
    """
    Cast a data object as a tensor.

    Parameters
    ----------
    dtype : torch datatype (default torch.FloatTensor)
        Output tensor datatype.
    dim : int or 'auto' (default 'auto')
        Minimum dimension of the output tensor. If this is 'auto', then no
        adjustments are made to the output tensor dimension. Otherwise, a
        singleton dimension is iteratively added until the tensor dimension
        is greater than or equal to the specified argument.
    """
    def __init__(self, dtype='torch.FloatTensor', dim='auto'):
        self.dim = dim
        self.dtype = dtype

    def __call__(self, data):
        return F.to_tensor(data, dtype=self.dtype, dim=self.dim)

    def __repr__(self):
        return f'{type(self).__name__}(dtype={self.dtype}, dim={self.dim})'


class ToNamedTensor(ToTensor):
    """
    Cast a data object as a named tensor.

    Parameters
    ----------
    dtype : torch datatype (default torch.FloatTensor)
        Output tensor datatype.
    dim : int or 'auto' (default 'auto')
        Minimum dimension of the output tensor. If this is 'auto', then no
        adjustments are made to the output tensor dimension. Otherwise, a
        singleton dimension is iteratively added until the tensor dimension
        is greater than or equal to the specified argument.
    names : list(str) or None (default None)
        List of names to assign the dimensions of the produced tensor.
    truncate: 'last' or 'first' (default 'last')
        Truncation rule if the tensor has fewer nameable dimensions than the
        transform has names. If 'last', excess names are truncated from the
        end; if 'first', they are truncated from the beginning.
    """
    def __init__(self, dtype='torch.FloatTensor', dim='auto',
                 names=None, truncate='last'):
        self.all_names = names
        self.truncate = truncate
        super(ToNamedTensor, self).__init__(dtype, dim)

    def _names(self, data):
        check = torch.Tensor(data)
        if self.truncate == 'last':
            return self.all_names[:check.dim()]
        elif self.truncate == 'first':
            return self.all_names[-check.dim():]

    def __call__(self, data):
        names = self._names(data)
        return F.to_named_tensor(data, dtype=dtype, names=names)

    def __repr__(self):
        return f'{type(self).__name__}(dtype={self.dtype}, dim={self.dim})'


class NaNFill(object):
    """
    Populate missing values marked by NaN entries.

    Parameters
    ----------
    fill : float, None, or Tensor/ndarray (default None)
        Value(s) used to replace NaN entries in the input tensor. If this is
        None, then this operation does nothing. If this is a tensor, it must
        be broadcastable against the missing part of the input tensor.
    """
    def __init__(self, fill=None):
        self.fill = fill

    def __call__(self, data):
        return F.nanfill(data, fill=self.fill)

    def __repr__(self):
        return f'{type(self).__name__}(fill={self.fill})'


class ApplyModelSpecs(object):
    """
    Transform an input dataset according to a specified collection of models.

    Each model specification is used to select and transform a subset of data
    columns and thereby produce a new DataFrame containing the specified
    model. Metadata can optionally be provided to guide column selection; for
    some use cases, it might be compulsory. The output is a dictionary mapping
    each model's name to its corresponding DataFrame.

    Parameters
    ----------
    models : list(ModelSpec)
        List of ModelSpec objects specifying the models to be produced from
        the input data and metadata.
    """
    def __init__(self, models):
        self.models = models

    def __call__(self, data, metadata=None):
        return F.apply_model_specs(
            models=self.models, data=data, metadata=metadata)

    def __repr__(self):
        s = f'{type(self).__name__}(models=['
        for m in self.models:
            s += indent(f'\n{m}', INDENT)
        if self.models:
            s += '\n'
        s += '])'
        return s


class ApplyTransform(object):
    """
    Apply a transformation to each value in an iterable.

    If the iterable is a dictionary, each value is transformed; if it is a
    list, each entry is transformed. The transformation is not recursive; for
    a recursive transformation on a list (potentially of lists), use
    `transform_block`.

    Parameters
    ----------
    transform : callable
        Transform to apply.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, iterable):
        return F.apply_transform(iterable, transform=self.transform)

    def __repr__(self):
        return f'{type(self).__name__}(transform={self.transform})'


class BlockTransform(object):
    """
    Apply a transformation to each entry in a list block. Only the base
    entries are transformed.

    Parameters
    ----------
    transform : callable
        Transform to apply.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, block):
        return F.transform_block(block, transform=self.transform)

    def __repr__(self):
        return f'{type(self).__name__}(transform={self.transform})'


class UnzipTransformedBlock(object):
    """
    Change a list block of dictionaries to a dictionary of list blocks.
    """
    #def __init__(self, depth=-1):
    #    self.depth = depth
    #    self.cur_depth = 0

    def __call__(self, block):
        return F.unzip_blocked_dict(block)

    def __repr__(self):
        return f'{type(self).__name__}()'


class ConsolidateBlock(object):
    """
    Consolidate a list block of tensors into a single tensor. If the tensors
    in the list block are of different sizes, smaller tensors are padded with
    NaN entries until all sizes are conformant before consolidation.
    """
    def __call__(self, block):
        return F.consolidate_block(block)

    def __repr__(self):
        return f'{type(self).__name__}()'


class ReadDataFrame(object):
    """
    Load tabular data from the specified path. Defaults to TSV. Any additional
    parameters are forwarded as arguments to `pd.read_csv`.
    """
    def __init__(self, sep='\t', **kwargs):
        self.sep = sep
        self.kwargs = kwargs

    def __call__(self, path):
        return F.read_data_frame(path, sep=self.sep, **self.kwargs)

    def __repr__(self):
        return f'{type(self).__name__}()'


class ReadNeuroImage(object):
    """
    Load neuroimaging data from the specified path. Any additional parameters
    are forwarded as arguments to `nibabel.load`.
    """
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, path):
        return F.read_neuro_image(path, **self.kwargs)

    def __repr__(self):
        return f'{type(self).__name__}()'


class EncodeOneHot(object):
    """
    Encode a categorical variable as a one-hot vector.

    Note that all encodings are internally stored by the transformation.
    Because it can be expensive to store a large matrix, this is not a
    recommended way to create one-hot encodings for categorical variables
    with a very large number of levels.

    Parameters
    ----------
    n_levels : int
        Total number of levels of the categorical variable.
    dtype : torch datatype (default torch.FloatTensor)
        Output tensor datatype. Defaults to float for gradient support.
    """
    def __init__(self, n_levels, dtype='torch.FloatTensor'):
        self.n_levels = n_levels
        self.dtype = dtype
        self.patterns = torch.eye(self.n_levels)

    def __call__(self, data):
        return F.vector_encode(data, encoding=self.patterns, dtype=self.dtype)

    def __repr__(self):
        return f'{type(self).__name__}(n_levels={self.n_levels})'


class ReadJSON(object):
    """
    Load JSON-formatted metadata into a python dictionary.
    """
    def __call__(self, path):
        return F.read_json(path)

    def __repr__(self):
        return f'{type(self).__name__}()'


class ChangeExtension(object):
    """
    Edit a path to change its extension.

    Parameters
    ----------
    new_ext : str
        New extension for the path.
    mode : 'all' or 'last' (default 'all')
        'all' indicates that everything after the first period in the base
        name is to be treated as the old extension and replaced; 'last'
        indicates that only text after the final period in the base name is to
        be treated as the old extension.
    """
    def __init__(self, new_ext, mode='all'):
        self.new_ext = new_ext
        self.mode = mode

    def __call__(self, path):
        return F.change_extension(path, new_ext=self.new_ext, mode=self.mode)

    def __repr__(self):
        return f'{type(self).__name__}(new_ext={self.new_ext})'


class NIfTIHeader(object):
    """
    Load metadata from a NIfTI header into a Python dictionary.
    """
    def __call__(self, path):
        return F.nifti_header(path)

    def __repr__(self):
        return f'{type(self).__name__}()'


class CWBCIfTIHeader(object):
    """
    Use connectome workbench to load metadata from a CIfTI header into a Python
    dictionary.
    """
    def __call__(self, path):
        return F.cwb_cifti_header(path)

    def __repr__(self):
        return f'{type(self).__name__}()'


class ToTensorX(DataObjectTransform, ToTensor):
    """
    `ToTensor` transformation applied to the assigned data of a
    DataObjectVariable. Consult `ToTensor` for additional details.
    """


class ToNamedTensorX(DataObjectTransform, ToNamedTensor):
    """
    `ToNamedTensor` transformation applied to the assigned data of a
    DataObjectVariable. Consult `ToNamedTensor` for additional details.
    """


class NaNFillX(DataObjectTransform, NaNFill):
    """
    `NaNFill` transformation applied to the assigned data of a
    DataObjectVariable. Consult `NaNFill` for additional details.
    """


class ApplyModelSpecsX(DataObjectTransformWithMetadata, ApplyModelSpecs):
    """
    `ApplyModelSpecs` transformation applied to the assigned data of a
    DataObjectVariable. Consult `ApplyModelSpecs` for additional details.
    """


class ReadDataFrameX(DataObjectTransform, ReadDataFrame):
    """
    `ReadDataFrame` transformation applied to the assigned data of a
    DataObjectVariable. Consult `ReadDataFrame` for additional details.
    """


class ReadNeuroImageX(DataObjectTransform, ReadNeuroImage):
    """
    `ReadNeuroImage` transformation applied to the assigned data of a
    DataObjectVariable. Consult `ReadNeuroImage` for additional details.
    """


class ReadJSONX(DataObjectTransform, ReadJSON):
    """
    `ReadJSON` transformation applied to the assigned data of a
    DataObjectVariable. Consult `ReadJSON` for additional details.
    """


class ChangeExtensionX(DataObjectTransform, ChangeExtension):
    """
    `ChangeExtension` transformation applied to the assigned data of a
    DataObjectVariable. Consult `ChangeExtension` for additional details.
    """


class DumpX(object):
    """
    Dump the data assignment content of a DataObjectVariable for further
    processing. Metadata references are lost and any steps that require
    metadata should be executed before this transform in a composition.
    The result can be transformed directly to a tensor.
    """
    def __call__(self, data):
        return F.dump_data(data)

    def __repr__(self):
        return f'{type(self).__name__}()'


class MetadataKeyX(object):
    """
    Obtain a value from the metadata block of an assigned DataObjectVariable.

    Parameters
    ----------
    key : hashable
        Variable whose value is obtained from the metadata block.
    """
    def __init__(self, key):
        self.key = key

    def __call__(self, data):
        return F.get_metadata_variable(data, self.key)

    def __repr__(self):
        return f'{type(self).__name__}({self.key})'
