# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Minimal MSC dataset
~~~~~~~~~~~~~~~~~~~
Minimal MSC dataset for testing or rapid prototyping.
"""
from __future__ import annotations
import os
import pathlib
import re
import shutil
import zipfile
from typing import Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import pandas as pd
import requests
from pkg_resources import resource_filename as pkgrf

from ..engine import extend_to_max_size


def get_msc_dir() -> str:
    output = pkgrf(
        'hypercoil',
        'neuro/datasets',
    )
    return f'{output}/MSCMinimal'


def minimal_msc_download(delete_if_exists: bool = False) -> str:
    output = pkgrf(
        'hypercoil',
        'neuro/datasets',
    )
    if delete_if_exists:
        shutil.rmtree(f'{output}/MSCMinimal', ignore_errors=True)
    # TODO: unsafe. No guarantee that the right files are inside the directory.
    if os.path.isdir(f'{output}/MSCMinimal'):
        return f'{output}/MSCMinimal'

    os.makedirs(output, exist_ok=True)

    # TODO: not a long-term solution. Ideally we'd do this the lazy way, but
    # this isn't a priority right now.
    url = (
        'https://github.com/hypercoil/miniMSC/archive/'
        '4b346577c67e278e0f51b209a83aed87811788c6.zip'
    )
    r = requests.get(url, allow_redirects=True)
    zip_target = f'{output}/MSCMinimal.zip'
    with open(zip_target, 'wb') as f:
        f.write(r.content)
    with zipfile.ZipFile(zip_target, 'r') as zip_ref:
        zip_ref.extractall(f'{output}')
    shutil.move(
        f'{output}/miniMSC-4b346577c67e278e0f51b209a83aed87811788c6',
        f'{output}/MSCMinimal',
    )
    os.remove(zip_target)
    return f'{output}/MSCMinimal'


def minimal_msc_all_pointers(
    dir: Optional[str] = None,
    sub: Optional[str] = None,
    task: Optional[str] = None,
    ses: Optional[str] = None,
) -> Sequence[Tuple[str, str]]:
    if dir is None:
        dir = get_msc_dir()
    if sub is None:
        sub = '*'
    if ses is None:
        ses = '*'
    if task is None:
        task = '*'
    paths = pathlib.Path(f'{dir}/data/ts/').glob(
        f'sub-MSC{sub}_ses-func{ses}_task-{task}_*ts.1D'
    )
    paths = list(paths)
    paths.sort()
    paths = tuple(
        (
            str(path),
            f"{re.sub(r'/ts/', '/motion/', str(path))[:-17]}relRMS.1D",
        )
        for path in paths
    )
    return paths


class MSCMinimal:
    """
    Minimal version of the Midnight Scan Club dataset.
    """

    # TODO: add option to split by subject or session
    # TODO: gotta implement DataLoader / torch data compatibility for
    # parallelisation. or use async.
    def __init__(
        self,
        delete_if_exists: bool = False,
        # all_in_memory: bool = False,
        shuffle: bool = True,
        batch_size: int = 1,
        sub: Optional[str] = None,
        task: Optional[str] = None,
        ses: Optional[str] = None,
        rms_thresh: float = None,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        self.srcdir = minimal_msc_download(delete_if_exists=delete_if_exists)
        self.paths = minimal_msc_all_pointers(
            self.srcdir, sub=sub, task=task, ses=ses
        )
        self.max = len(self.paths)
        self.shuffle = shuffle
        self.rms_thresh = rms_thresh
        self.batch_size = batch_size
        self.cfg_iter(shuffle=shuffle, key=key)

    def _load_single(self, idx):
        # As of now, we force all proposed indices into the valid range
        if idx >= self.max:
            idx = self.max
        idx = self.idxmap[idx]
        bold, motion = self.paths[idx]
        bold = pd.read_csv(bold, sep=' ', header=None)
        ret = {
            '__id__': self.paths[idx][0].split('/')[-1][:-18],
            'bold': jnp.array(bold.values.T.astype(jnp.float32)),
        }
        if self.rms_thresh is not None:
            motion = pd.read_csv(motion, sep=' ', header=None)
            tmask = motion <= self.rms_thresh
            ret.update(
                {
                    'tmask': jnp.array(tmask.values.T.astype(bool)),
                }
            )
        return ret

    def _load_batch(self):
        batchref = tuple(
            self._load_single(idx)
            for idx in range(self.idx, self.idx + self.batch_size)
        )
        batch = {
            '__id__': tuple(item['__id__'] for item in batchref),
            'bold': jnp.stack(
                extend_to_max_size(
                    tuple(item['bold'] for item in batchref), fill=0
                )
            ),
        }
        if self.rms_thresh is not None:
            batch.update(
                {
                    'tmask': jnp.stack(
                        extend_to_max_size(
                            tuple(item['tmask'] for item in batchref),
                            fill=False,
                        )
                    ),
                }
            )
        return batch

    def cfg_iter(
        self,
        shuffle: bool,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        self.idx = -self.batch_size
        idxmap = jnp.arange(self.max)
        if shuffle:
            idxmap = jax.random.permutation(
                key=key, x=idxmap, independent=True
            )
        self.idxmap = idxmap

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += self.batch_size
        if self.idx < self.max:
            return self._load_batch()
        raise StopIteration
