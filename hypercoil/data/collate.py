# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Collate functions
~~~~~~~~~~~~~~~~~
Patch of pytorch default_collate and associated functionality that enables
additional arguments.
"""

#TODO
# Keeping this in sync with pytorch could be a nuisance and a failure point.
# Update for autonomy.

import torch
import re
import collections
from torch._six import string_classes
from .functional import extend_to_max_size

np_str_obj_array_pattern = re.compile(r'[SaUO]')


int_classes = int


gen_collate_err_msg_format = (
    "gen_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def gen_collate(
    batch,
    concat=torch.stack,
    concat_axis=0
):
    """
    Basically stolen/adapted from torch/utils/data/_utils/collate.py
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage).resize_(len(batch), *list(elem.size()))
        return concat(batch, concat_axis, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(gen_collate_err_msg_format.format(elem.dtype))

            return gen_collate([torch.as_tensor(b) for b in batch],
                               concat=concat,
                               concat_axis=concat_axis)
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        try:
            return elem_type({
                key: gen_collate([d[key] for d in batch],
                                 concat=concat,
                                 concat_axis=concat_axis)
                for key in elem
            })
        except TypeError:
            # The mapping type may not support `__init__(iterable)`.
            return {key: gen_collate([d[key] for d in batch],
                                     concat=concat,
                                     concat_axis=concat_axis)
                    for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(gen_collate(samples,
                                       concat=concat,
                                       concat_axis=concat_axis)
                           for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError(
                'each element in list of batch should be of equal size')
        # It may be accessed twice, so we use a list.
        transposed = list(zip(*batch))

        if isinstance(elem, tuple):
            # Backwards compatibility.
            return [gen_collate(samples,
                                concat=concat,
                                concat_axis=concat_axis)
                    for samples in transposed]
        else:
            try:
                return elem_type([
                    gen_collate(samples,
                                concat=concat,
                                concat_axis=concat_axis)
                    for samples in transposed
                ])
            except TypeError:
                # The sequence type may not support `__init__(iterable)`
                # (e.g., `range`).
                return [gen_collate(samples,
                                    concat=concat,
                                    concat_axis=concat_axis)
                        for samples in transposed]

    raise TypeError(gen_collate_err_msg_format.format(elem_type))


def extend_and_bind(tensors, axis=0, out=None):
    tensors = extend_to_max_size(tensors)
    size_1 = [t.size(axis) == 1 for t in tensors]
    if all(size_1):
        return torch.cat(tensors, dim=axis, out=out)
    return torch.stack(tensors, dim=axis, out=out)
