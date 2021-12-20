import random, pathlib, tarfile, shutil, warnings
import torch
import webdataset
from collections import OrderedDict
from functools import partial
from webdataset.autodecode import torch_loads
from hypercoil.data import functional as F
from hypercoil.data.collate import gen_collate, extend_and_bind
from hypercoil.data.dataset import ReferencedDataset

"""
ref = 'hcp_full'
if ref == 'bids':
    from hypercoil.data.bids import fmriprep_references
    data_dir = '/mnt/pulsar/Research/Development/hypercoil/hypercoil/examples/ds-synth/'
    layout, images, confounds, refs = fmriprep_references(
        data_dir,
        model=['(dd1(rps + wm + csf + gsr))^^2']
    )
    df = layout.dataset(
        observations=('subject',),
        levels=('task', 'session', 'run')
    )


if ref == 'hcp':
    from hypercoil.data.hcp import hcp_references
    data_dir = '/mnt/andromeda/Data/HCP_subsubsample/'
    layout, images, confounds, refs = hcp_references(
        data_dir,
        model=['(dd1(rps + wm + csf + gsr))^^2'],
        tmask='and(1_[framewise_displacement < 0.2], 1_[std_dvars < 2.5])'
    )
    df = layout.dataset(
        observations=('subject',),
        levels=('task', 'session', 'run')
    )


if ref == 'hcp_full':
    from hypercoil.data.hcp import hcp_references
    data_dir = '/mnt/andromeda/Data/HCP_S1200/'
    layout, images, confounds = hcp_references(
        data_dir,
        model=['(dd1(rps + wm + csf + gsr))^^2'],
        tmask='and(1_[framewise_displacement < 0.2], 1_[std_dvars < 2.5])'
    )
    df = layout.dataset(
        observations=('subject',),
        levels=('task', 'session', 'run')
    )


ds = ReferencedDataset(refs, lambda x: x)
ds.set_depth(1)
refs = ds.data_refs
"""

def split_refs(refs, n_splits, split_level, shuffle_split, seed=None):
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
    try:
        data = ref()
    except AttributeError:
        warnings.warn(f'Discarding broken reference: {ref.ids}')
        return
    basename = '_'.join([f'{k}-{v}' for k, v in ref.ids.items()])
    save_data(dirname, basename, data, baseext=None)


def save_data(dirname, basename, data, baseext=None):
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
    tar_path = f'{dirname}.tar'
    with tarfile.open(tar_path, 'w:') as tar:
        tar.add(dirname, arcname='.')


def torch_wds(path, keys, map, shuffle=100, batch_size=1):
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
    seed=None):

    wds_path = path or f'/tmp/wds-{random.getrandbits(128)}'
    pathlib.Path(wds_path).mkdir(parents=True, exist_ok=True)
    if shard_size and n_shards:
        raise ValueError('Cannot specify both shard size and number')


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
            print(f'[SPLIT {i:03d}] [SHARD {j:06d}]')
            make_shard(dirname)
            shutil.rmtree(dirname)

    n_train = max(n_splits - 2, 1)
    ds_path = (f'{wds_path}/spl-{{000..{n_train:03d}}}_'
               f'shd-{{000000..{(n_shards - 1):06d}}}.tar')
    wds = torch_wds(ds_path)
    return wds

"""
wds_path = '/mnt/andromeda/Data/HCP_wds'
shuffle_shard = True
shuffle_split = True
split_level = ('subject',)
n_splits = 5
shard_level = ()
n_shards = 4
shard_size = 0
seed = None


wds = make_wds(
    refs=refs,
    path=wds_path,
    n_splits=n_splits,
    n_shards=n_shards,
    shuffle_shard=shuffle_shard,
    shuffle_split=shuffle_split,
    split_level=split_level,
    shard_level=shard_level,
    shard_size=shard_size,
    seed=seed)
"""
