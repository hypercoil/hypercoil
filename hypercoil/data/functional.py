# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Functions for transforming various data modalities.

.. warning::
    Data transformations are, at the moment, largely undocumented and
    untested. This will change during a coming round of development; however,
    in the interim, users are advised to verify that all outputs are as
    expected.
"""
import re, json, subprocess
import bs4
import lxml
import torch
import numpy as np
import pandas as pd
import nibabel as nb
from functools import reduce, partial


def identity(data):
    """Return the input."""
    return data


def get_col(df, label):
    """
    Obtain the data or index column with the given label from a DataFrame.
    """
    try:
        return df.index.get_level_values(label)
    except KeyError:
        return df[label]


def to_tensor(data, dtype=None, device=None, dim='auto'):
    """
    Change the input data into a tensor with the specified type and minimum
    dimension.

    Parameters
    ----------
    data : object
        Object that can be cast to a Tensor. If this is a pandas DataFrame,
        only the values are cast to a Tensor.
    dtype : torch datatype
        Output tensor datatype.
    dim : int or 'auto' (default 'auto')
        Minimum dimension of the output tensor. If this is 'auto', then no
        adjustments are made to the output tensor dimension. Otherwise, a
        singleton dimension is iteratively added until the tensor dimension
        is greater than or equal to the specified argument.

    Returns
    -------
    tensor : Tensor
        PyTorch Tensor containing the input data.
    """
    if isinstance(data, pd.DataFrame):
        #TODO: transposing as a patch here. Move it somewhere reasonable.
        data = data.values.T
    try:
        tensor = torch.tensor(data, dtype=dtype, device=device)
    except TypeError:
        tensor = torch.Tensor([data], dtype=dtype, device=device)
    if dim != 'auto':
        current_dim = tensor.dim()
        deficit = dim - current_dim
        if deficit < 0:
            raise ValueError(
                f'Requested an output tensor of dimension {dim}, but '
                f'cannot reduce tensor dimension below {current_dim}')
        elif deficit > 0:
            expansion = [...] + [None for _ in range(deficit)]
            tensor = tensor[expansion]
    return tensor


def to_named_tensor(data, dtype=None, device=None, dim='auto', names=None):
    """
    Change the input data into a named tensor with the specified type and
    minimum dimension.

    Parameters
    ----------
    data : object
        Object that can be cast to a Tensor. If this is a pandas DataFrame,
        only the values are cast to a Tensor.
    dtype : torch datatype
        Output tensor datatype.
    dim : int or 'auto' (default 'auto')
        Minimum dimension of the output tensor. If this is 'auto', then no
        adjustments are made to the output tensor dimension. Otherwise, a
        singleton dimension is iteratively added until the tensor dimension
        is greater than or equal to the specified argument.
    names : list(str) or None (default None)
        List of names to assign the dimensions of the produced tensor.

    Returns
    -------
    tensor : Tensor
        PyTorch Tensor containing the input data.
    """
    out = to_tensor(data, dtype=dtype, device=device, dim=dim)
    out.names = names
    return out


def nanfill(data, fill=None):
    """
    Populate all missing or invalid values in a tensor.

    Parameters
    ----------
    data : Tensor or np.ndarray
        Tensor that potentially contains missing or invalid data denoted as
        NaN.
    fill : float, None, or Tensor/ndarray (default None)
        Value(s) used to replace NaN entries in the input tensor. If this is
        None, then this operation does nothing. If this is a tensor, it must
        be broadcastable against the missing part of the input tensor. If this
        is the string 'mean', then every NaN entry is replaced with the mean
        of non-NaN entries.

    Returns
    -------
    data : Tensor or np.ndarray
        Input tensor with missing or invalid values populated.
    nanmask : Tensor or np.ndarray
        Boolean tensor indicating the cells where NaN entries were previously
        located (returned so that they can be restored later if desired).
    """
    #TODO: enable interpolation from nearest neighbours along set axes
    nanmask = None
    if fill is not None:
        if isinstance(data, torch.Tensor):
            nanmask = torch.isnan(data)
            if fill == 'mean':
                fill = data.nanmean()
            data = torch.nan_to_num(data, fill)
        elif isinstance(data, np.ndarray):
            nanmask = np.isnan(data)
            if fill == 'mean':
                fill = np.nanmean(data)
            data = np.nan_to_num(data, fill)
        else:
            raise TypeError(
                'nanfill: unrecognised input type. '
                'Provide either a tensor or a numpy array as input.'
            )
    return data, nanmask


def apply_model_specs(models, data, metadata=None):
    """
    Transform an input dataset according to a specified collection of models.

    Each model specification is used to select and transform a subset of data
    columns and thereby produce a new DataFrame containing the specified
    model. Metadata can optionally be provided to guide column selection; for
    some use cases, it might be compulsory. The output is a dictionary mapping
    each model's name to its corresponding DataFrame.
    """
    if models is None:
        return {}
    return {m.name: m(data, metadata=metadata) for m in models}


def apply_transform(iterable, transform):
    """
    Apply a transformation to each value in an iterable.

    If the iterable is a dictionary, each value is transformed; if it is a
    list, each entry is transformed. The transformation is not recursive; for
    a recursive transformation on a list (potentially of lists), use
    `transform_block`.
    """
    if isinstance(iterable, dict):
        return {k: transform(v) for k, v in iterable.items()}
    else:
        return [transform(v) for v in iterable]


def apply_to_select(iterable, transform, selection=None):
    """
    Apply a transformation to selected values in an iterable.

    If the iterable is a dictionary, each key-specified value is transformed;
    if it is a list, each index-specified entry is transformed.
    """
    if isinstance(iterable, dict):
        return {k: transform(v) if k in selection else v
                for k, v in iterable.items()}
    else:
        return [transform(v) if i in selection else v
                for i, v in enumerate(iterable)]


def transform_block(block, transform):
    """
    Apply a transformation to each entry in a list block. Only the base
    entries are transformed.
    """
    if isinstance(block, list):
        return [transform_block(e, transform) for e in block]
    return transform(block)


def unzip_blocked_dict(block):
    """
    Change a list block of dictionaries to a dictionary of list blocks.
    """
    if isinstance(block[0], list):
        return unzip_blocked_dict([unzip_blocked_dict(e) for e in block])
    return {k: [d[k] for d in block] for k in block[0].keys()}


def consolidate_block(block):
    """
    Consolidate a list block of tensors into a single tensor. If the tensors
    in the list block are of different sizes, smaller tensors are padded with
    NaN entries until all sizes are conformant before consolidation.
    """
    if isinstance(block, torch.Tensor):
        return block
    elif len(block) == 1:
        return consolidate_block(block[0])
    elif isinstance(block[0], torch.Tensor):
        out = extend_to_max_size(block)
    else:
        out = extend_to_max_size([consolidate_block(e) for e in block])
    out = torch.stack(out)
    #TODO: stupid patch here to skip leading singleton dimension.
    if out.size(0) == 1:
        return out.squeeze(0)
    return out


def consolidate_to_tensor(block, dtype=None, device=None, dim='auto'):
    """
    Convenience function for ReferencedDataset because pydra doesn't always
    play well with tensors. Wraps `consolidate_block` and `to_tensor` while
    also handling subdictionaries if they exist.
    """
    if isinstance(block, dict):
        return {
            k: consolidate_to_tensor(v, dtype=dtype, device=device, dim=dim)
            for k, v in block.items()
        }
    tt = partial(to_tensor, dtype=dtype, device=device, dim=dim)
    block = transform_block(block, tt)
    return consolidate_block(block)



def ravel(lst, stack=[], max_depth=None):
    try:
        dim = reduce(lambda x, y: x * y, stack)
    except TypeError:
        dim = 1
    max_depth = max_depth or float('inf')
    depth = len(stack)
    while depth < max_depth:
        stack += [len(lst) // dim]
        if not isinstance(lst[0], list):
            break
        lst = [item for sublist in lst for item in sublist]
        dim *= stack[-1]
        depth = len(stack)
    return lst, stack


def fold(lst, stack):
    while len(stack) > 0:
        dim = stack.pop()
        lst = list(zip(*[iter(lst)] * dim))
    return lst[0], stack


def extend_to_max_size(tensor_list):
    """
    Extend all tensors in a list until their sizes are equal to the size
    of the largest tensor along each axis. Any new entries created via
    extension are marked with NaN to denote that they were missing from
    the input; they can be populated by chaining this with a call to
    `nanfill`.
    """
    sizes = [t.size() if t.dim() != 0 else [1] for t in tensor_list]
    max_size = torch.amax(
        torch.tensor(sizes, dtype=torch.long, device=tensor_list[0].device),
        0
    )
    return [extend_to_size(t, max_size) for t in tensor_list]


def extend_to_size(tensor, size):
    """
    Extend a tensor in the positive direction until its size matches the
    specification. Any new entries created via extension are marked with
    NaN to denote that they were missing from the input; they can be
    populated by chaining this with a call to `nanfill`.
    """
    out = torch.empty(
        *size,
        dtype=tensor.dtype,
        device=tensor.device
    ) * float('nan')
    #TODO: revisit named tensors when they are stable
    #names = tensor.names
    #tensor = tensor.rename(None)
    out[[slice(s) for s in tensor.size()]] = tensor
    #out.names = names
    return out


def get_path_from_var(var):
    """
    Ensure that the provided variable is a string representing a path on the
    filesystem.
    """
    try:
        return str(var.path)
    except AttributeError:
        return str(var)


def read_data_frame(path, sep='\t', **kwargs):
    """
    Load tabular data from the specified path. Defaults to TSV. Any additional
    arguments are forwarded to `pd.read_csv`.
    """
    path = get_path_from_var(path)
    return pd.read_csv(path, sep=sep, **kwargs)


def read_neuro_image(path, **kwargs):
    """
    Load neuroimaging data from the specified path. Any additional arguments
    are forwarded to `nibabel.load`.
    """
    path = get_path_from_var(path)
    img = nb.load(path, **kwargs)
    if isinstance(img, nb.Cifti2Image):
        #TODO: should check the matrix axes for the time dimension and flip it
        # to the end. Right now instead we assume HCP-style CIfTI input,
        # greyordinates x time
        return img.get_fdata().transpose(-1, -2)
    return img.get_fdata()


def vector_encode(data, encoding, device=None):
    """
    Encode a categorical variable as a vector.

    Parameters
    ----------
    data : object
        Object that can be cast to a Tensor. Each entry represents the value
        of a categorical variable.
    encoding : Tensor
        Tensor whose indices return an encoding for each level of the
        categorical variable. If this is an identity matrix whose dimension
        equals the number of levels of the categorical variable, then this
        represents a one-hot endoding. Note that because it can be expensive
        to store a large matrix, this is not a recommended way to create
        one-hot encodings for categorical variables with many levels.
    """
    if device is None and isinstance(data, torch.Tensor):
        device = data.device
    idx = torch.tensor(data, dtype=torch.long, device=device)
    return encoding[idx]


def change_extension(path, new_ext, mode='all'):
    """
    Edit a path to change its extension.

    Parameters
    ----------
    path : str
        Path on a file system.
    new_ext : str
        New extension for the path.
    mode : 'all' or 'last' (default 'all')
        'all' indicates that everything after the first period in the base
        name is to be treated as the old extension and replaced; 'last'
        indicates that only text after the final period in the base name is to
        be treated as the old extension.

    Returns
    -------
    new_path : str
        Input path with the specified new extension.
    """
    if mode == 'all':
        return re.sub(r'(\.)[^\/]*$', f'\\1{new_ext}', path)
    elif mode == 'last':
        return re.sub(r'(\.)[^\/\.]*$', f'\\1{new_ext}', path)


def read_json(path):
    """
    Load JSON-formatted metadata into a python dictionary.

    Parameters
    ----------
    path : str
        Path to the JSON-formatted metadata file.

    Returns
    -------
    metadata : dict
        Python dictionary containing all metadata in the JSON file.
    """
    with open(path) as file:
        metadata = json.load(file)
    return metadata


def nifti_header(path):
    """
    Load some of the most essential (TM) metadata from a NIfTI file's header
    into a python dictionary. Right now, this means the TR.

    Note that this operation is not smart enough to figure out the units if
    they're not in the header. It will also assume that the fourth dimension is
    time when setting repetition time. If it's not, you probably don't need
    a repetition time anyway.

    Path
    ----
    Path to the NIfTI file.

    Returns
    -------
    metadata : dict
        Python dictionary containing all metadata in the NIfTI header.
    """
    metadata = {}
    hdr = nb.load(path).header
    try: # standard NIfTI case
        t_rep = hdr['pixdim'][4]
        t_units = hdr.get_xyzt_units()[1]
        if t_units in ('s', 'sec', 'seconds'):
            pass
        elif t_units in ('ms', 'msec', 'millisec', 'milliseconds'):
            t_rep /= 1000.
        else:
            raise RuntimeError(f'Unrecognised units: {t_units}')
        metadata['RepetitionTime'] = t_rep
    except TypeError: # CIfTI case
        t_rep = [
            ax.series_step for ax in hdr.matrix if ax.series_unit == 'SECOND'
        ]
        if len(t_rep) != 1:
            raise RuntimeError(
                f'Conflicting or missing information for TR: {t_rep}'
            )
        metadata['RepetitionTime'] = t_rep[0]
    try:
        metadata['RepetitionTime']
    except KeyError:
        raise RuntimeError(f'Failed to find required field: repetition time')
    return metadata


def cwb_cifti_header(path):
    """
    Nibabel is rather slow when it comes to parsing CIfTI data. To limit the
    number of times we have to do this, we can use connectome workbench if it's
    available to obtain header information from a CIfTI file. This isn't as
    generalisable as `nifti_header`, but it should be more efficient for large
    CIfTI datasets so long as connectome workbench is available.
    """
    metadata = {}
    cmd = f'wb_command -nifti-information {path} -print-xml|grep SECOND'
    data = bs4.BeautifulSoup(subprocess.check_output(cmd, shell=True), 'lxml')
    metadata['RepetitionTime'] = float(
        data.find('matrixindicesmap').get('seriesstep')
    )
    return metadata


def dump_data(dataobj):
    """
    Return only the data block of an assigned DataObjectVariable.
    """
    return dataobj.data


def get_metadata_variable(dataobj, key):
    """
    Obtain a value from the metadata block of an assigned DataObjectVariable.
    """
    return dataobj.metadata.get(key)


def polynomial_detrend(tensor, order=0):
    """Apply a polynomial detrend of the specified order to the data."""
    base = torch.linspace(
        0, 1, tensor.size(-1),
        dtype=tensor.dtype,
        device=tensor.device
    )
    X = torch.zeros(
        (tensor.size(-1), order + 1),
        dtype=tensor.dtype,
        device=tensor.device
    )
    for o in range(order + 1):
        X[:, o] = base ** o
    betas = torch.linalg.pinv(X.T @ X) @ X.T @ tensor.transpose(-1, -2)
    return tensor - betas.transpose(-1, -2) @ X.T


def transpose(data):
    return data.transpose(-1, -2)


def standardise(data, axis=-1):
    #TODO: confusing: standardise for np arrays, normalise for tensors . . .
    # but then, the whole functional/data transform module needs a major
    # cleanup / revamp
    mu = np.nanmean(data, axis=axis, keepdims=True)
    std = np.nanstd(data, axis=axis, keepdims=True)
    return (data - mu) / std


def normalise(data, axis=-1):
    mu = data.nanmean(axis=axis, keepdims=True)
    #TODO: replace with nanstd when torch adds support
    # Right now, this is biased, so it won't give output with unbiased SD 1
    std = torch.nanmean(
        (data - mu) ** 2,
        axis=axis,
        keepdims=True
    ).sqrt()
    return (data - mu) / std


def window(data, window_length, window_start=0):
    window_end = window_start + window_length
    return data[..., window_start:window_end]


def random_window(data, window_length, axis=-1):
    #TODO: enable random (integer) length sampled from distribution
    # and non-uniform distribution for sampling start
    end = data.shape[axis]
    if window_length is None:
        return (end, 0)
    maximum = max(end - window_length, 1)
    window_start = np.random.randint(maximum)
    return (window_length, window_start)


def window_map(data, keys, window_length):
    window_length, window_start = random_window(data[keys[0]], window_length)
    return {
        k: (window(v, window_length, window_start) if k in keys else v)
        for k, v in data.items()
    }


def fillnan(data, nanmask):
    data[nanmask] = float('nan')
    return data
