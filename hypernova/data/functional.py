# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data transform functions
~~~~~~~~~~~~~~~~~~~~~~~~
Functions for transforming various data modalities.
"""
import re, json
import torch
import pandas as pd
import nibabel as nb


def to_tensor(data, dtype='torch.FloatTensor'):
    if isinstance(data, pd.DataFrame):
        data = data.values
    return torch.Tensor(data).type(dtype)


def to_named_tensor(data, dtype='torch.FloatTensor', names=None):
    out = to_tensor(data, dtype)
    out.names = names
    return out


def nanfill(data, fill=None):
    if fill is not None:
        if isinstance(data. np.ndarray):
            data[np.isnan(data)] = fill
        elif isinstance(data, torch.Tensor):
            data[torch.isnan(data)] = fill
    return data


def apply_model_specs(models, data, metadata=None):
    return {m.name: m(data, metadata=metadata) for m in models}


def apply_transform(iterable, transform):
    if isinstance(iterable, dict):
        return {k: transform(v) for k, v in iterable.items()}
    else:
        return [transform(v) for v in iterable]


def transform_block(block, transform):
    if isinstance(block, list):
        return [transform_block(e, transform) for e in block]
    return transform(block)


def unzip_blocked_dict(block):
    if isinstance(block[0], list):
        return unzip_blocked_dict([unzip_blocked_dict(e) for e in block])
    return {k: [d[k] for d in block] for k in block[0].keys()}


def consolidate_block(block):
    if isinstance(block, torch.Tensor):
        return block
    elif isinstance(block[0], torch.Tensor):
        out = extend_and_conform(block)
    else:
        out = extend_and_conform([consolidate_block(e) for e in block])
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


def get_path_from_var(var):
    try:
        return str(var.path)
    except AttributeError:
        return str(var)


def read_data_frame(path, sep='\t', **kwargs):
    path = get_path_from_var(path)
    return pd.read_csv(path, sep=sep, **kwargs)


def read_neuro_image(path, **kwargs):
    path = get_path_from_var(path)
    return nb.load(path, **kwargs).get_fdata()


def vector_encode(data, encoding, dtype='torch.FloatTensor'):
    idx = torch.Tensor(data).type('torch.LongTensor')
    return encoding[idx].type(dtype)


def change_extension(path, new_ext, mode='all'):
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


def dump_data(dataobj):
    return dataobj.data


def get_metadata_variable(dataobj, key):
    return dataobj.metadata[key]
