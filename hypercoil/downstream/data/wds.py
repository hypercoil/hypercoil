# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
WebDataset
~~~~~~~~~~
DataReference conversion to WebDataset format.
"""
import random, pathlib, tarfile, shutil, warnings
import torch
import webdataset
from collections import OrderedDict
from functools import partial
from webdataset.autodecode import torch_loads
from hypercoil.data.collate import gen_collate, extend_and_bind
from hypercoil.data.dataset import ReferencedDataset


def make_wds(
    refs,
    path,
    n_splits,
    n_shards,
    shuffle_shard=True,
    shuffle_split=True,
    split_level=(),
    shard_level=(),
    shard_size=0,
    seed=None
):
    """
    Create a `WebDataset` from a collection of `DataReference`s.

    Parameters
    ----------
    refs : iterable(DataReference)
        `DataReference`s to be saved into a `WebDataset`.
    path : str
        Write path for the `.tar` shards that constitute the `WebDataset`.
    n_splits : int
        Number of splits (equal subsets) into which the references are to be
        divided.
    n_shards : int
        Number of `.tar` shards to create per split as the `WebDataset`.
    shuffle_shard : bool (default True)
        Assigns data within each split to a random shard if true.
    shuffle_split : bool (default True)
        Results in random splits if true.
    split_level : iterable(ids) (default ())
        By default, this operation splits references into subsets without
        consideration for the identifiers of those references. This may lead
        to incorrect results in some cases. For instance, if multiple data
        points are collected per subject, they might be split across different
        data splits. If one of these points ends up in the test set, this
        could lead to overestimation of model performance. `split_level`
        allows specification of an identifier combination that configures the
        split. For instance, specifying ('subject',) as the split level will
        split across subjects instead of references (assuming the existence of
        a 'subject' field in each reference's internal ids). Specifying
        multiple ids will create a level for each existing combination of
        those ids' assignments in the dataset.
    shard_level : iterable(ids) (default ())
        As `split_level`, but constrains assignment of within-split references
        to shards.
    shard_size : int (default 0)
        Approximate number of `DataReference`s to include per shard. If this
        is assigned a nonzero value, `n_shards` must also be set to 0.
    seed : int or None (default None)
        Random seed for shuffling. Unused if `shuffle_split` is False.

    Returns
    -------
    Webdataset
        Webdataset resulting from consecutive splitting and sharding of
        `DataReference`s.
    """

    wds_path = path or f'/tmp/wds-{random.getrandbits(128)}'
    pathlib.Path(wds_path).mkdir(parents=True, exist_ok=True)
    if shard_size and n_shards:
        raise ValueError(
            'Cannot specify both shard size and number. '
            'Ensure `n_shards` is set to 0 to use the shard size.'
        )


    splits = split_refs(
        refs=refs,
        n_splits=n_splits,
        split_level=split_level,
        shuffle_split=shuffle_split,
        seed=seed
    )
    for i, split in enumerate(splits):
        n = len(split)
        if n_shards > n:
            n_shards = n
        max_shard_size = shard_size or -(-n // n_shards)
        shards = split_refs(
            refs=split,
            n_splits=n_shards,
            split_level=shard_level,
            shuffle_split=shuffle_shard,
            seed=seed
        )
        for j, shard in enumerate(shards):
            dirname = f'{wds_path}/spl-{i:03d}_shd-{j:06d}'
            for ref in shard:
                process_obs(ref=ref, dirname=dirname)
            print(f'[ SPLIT {i:03d} | SHARD {j:06d} ]')
            make_shard(dirname)
            shutil.rmtree(dirname)

    n_train = max(n_splits - 2, 1)
    ds_path = (f'{wds_path}/spl-{{000..{n_train:03d}}}_'
               f'shd-{{000000..{(n_shards - 1):06d}}}.tar')
    wds = torch_wds(ds_path)
    return wds


def torch_wds(path, keys, map, shuffle=100, batch_size=1):
    """
    Load a `WebDataset` from `.tar` shards.

    Parameter
    ---------
    path : str
        General path to the `WebDataset` shards to include in the WebDataset.
    keys : dict
        Dictionary specifying the elements of each data point in the dataset
        to include in each sample, together with the transformations to apply
        to each element. For instance,
        `{'x' : lambda x: x, 'y' : lambda y : EncodeOneHot()}`
        first identifies two elements, x and y, in each data sample. It then
        leaves x as is while encoding y as a one-hot vector.
    map : callable
        Transformation to apply to each observation as a whole. See `keys`
        for applying transformations to specific elements.
    shuffle : int (default 100)
        Size of data buffer. When the data loader creates a batch, it randomly
        chooses `batch_size` samples from the data buffer to constitute the
        batch. A larger buffer generally slows performance but increases
        randomness.
    batch_size : int (default 1)
        Batch size when sampling from the `WebDataset`.
    """
    #TODO: add arguments for n_splits_val and n_splits_test and automatically
    # form the general path based on the omission of these. But that should
    # probably be a parent function; leave this API exposed for those who want
    # a greater degree of control.
    collation_fn = partial(gen_collate, concat=extend_and_bind)
    return (
        webdataset.WebDataset(path)
        .shuffle(shuffle)
        .decode(lambda x, y: torch_loads(y))
        .map(map)
        .to_tuple(*keys.keys())
        .map_tuple(*keys.values())
        .batched(batch_size, collation_fn=collation_fn)
    )


def split_refs(refs, n_splits, split_level, shuffle_split, seed=None):
    """
    Split data references into approximately equal subsets.

    Parameters
    ----------
    refs : iterable(DataReference)
        Data references to be split into approximately equal subsets.
    n_splits : int
        Number of splits (equal subsets) into which the references are to be
        divided.
    split_level : iterable(ids) or None
        By default, this operation splits references into subsets without
        consideration for the identifiers of those references. This may lead
        to incorrect results in some cases. For instance, if multiple data
        points are collected per subject, they might be split across different
        data splits. If one of these points ends up in the test set, this
        could lead to overestimation of model performance. `split_level`
        allows specification of an identifier combination that configures the
        split. For instance, specifying ('subject',) as the split level will
        split across subjects instead of references (assuming the existence of
        a 'subject' field in each reference's internal ids). Specifying
        multiple ids will create a level for each existing combination of
        those ids' assignments in the dataset.
    shuffle_split : bool
        Results in random splits if true.
    seed : int or None (default None)
        Random seed for shuffling. Unused if `shuffle_split` is False.

    Returns
    -------
    list(list(DataReference))
        DataReferences split into `n_splits` approximately equal subsets.
    """
    random.seed(seed)
    levels = OrderedDict()
    if split_level:
        for ref in refs:
            key = tuple([ref.ids[l] for l in split_level])
            levels[key] = levels.get(key, []) + [ref]
    else:
        for key, ref in enumerate(refs):
            levels[key] = [ref]
    n_levels = len(levels)
    max_split_size = -(-n_levels // n_splits)
    n_max = n_levels % n_splits or n_splits
    asgt = []
    max_remain = n_max
    for i in range(n_splits):
        if max_remain:
            asgt += [i for _ in range(max_split_size)]
            max_remain -= 1
        else:
            asgt += [i for _ in range(max_split_size - 1)]
    if shuffle_split: random.shuffle(asgt)
    splits = [[] for _ in range(n_splits)]
    for i, l in enumerate(levels.values()):
        splits[asgt[i]] += l
    return splits


def process_obs(ref, dirname):
    """
    Process a single `DataReference` and save it to disk in torch format.
    """
    try:
        data = ref()
    except AttributeError:
        warnings.warn(f'Discarding broken reference: {ref.ids}')
        return
    basename = '_'.join([f'{k}-{v}' for k, v in ref.ids.items()])
    save_data(dirname, basename, data, baseext=None)


def save_data(dirname, basename, data, baseext=None):
    """
    Given a key-value dictionary representing different elements of a data
    sample (e.g., observations and labels), save the elements to disk in torch
    format, using the file extension to denote the data element.
    """
    baseext = baseext or ''
    for ext, val in data.items():
        if baseext:
            ext = '-'.join((baseext, ext))
        if isinstance(val, dict):
            save_data(dirname, basename, data, ext)
        path = f'{dirname}/{basename}.{ext}'
        pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        torch.save(val, path)


def make_shard(dirname):
    """
    Archive a directory in a `.tar` shard.
    """
    tar_path = f'{dirname}.tar'
    with tarfile.open(tar_path, 'w:') as tar:
        tar.add(dirname, arcname='.')
