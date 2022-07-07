# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Minimal datasets
~~~~~~~~~~~~~~~~
Minimal neuroimaging datasets for testing or rapid prototyping.
"""
import os
import re
import shutil
import pathlib
import random
import requests, zipfile
import torch
import pandas as pd
from pkg_resources import resource_filename as pkgrf
from ..data.functional import to_tensor, extend_to_max_size
from ..data.transforms import ToTensor


def minimal_msc_download(delete_if_exists=False):
    output = pkgrf(
        'hypercoil',
        'neuro/datasets'
    )
    if delete_if_exists:
        shutil.rmtree(f'{output}/MSCMinimal')
    #TODO: unsafe. No guarantee that the right files are inside the directory.
    if os.path.isdir(f'{output}/MSCMinimal'):
        return f'{output}/MSCMinimal'

    os.makedirs(output, exist_ok=True)

    #TODO: not a long-term solution. Ideally we'd do this the lazy way, but
    # there just isn't time now.
    url = (
        'https://github.com/rciric/xwave/archive/'
        'cf862c1dc3ead168db68abe4c870a509285fc729.zip'
    )
    r = requests.get(url, allow_redirects=True)
    zip_target = f'{output}/MSCMinimal.zip'
    with open(zip_target, 'wb') as f:
        f.write(r.content)
    with zipfile.ZipFile(zip_target, 'r') as zip_ref:
        zip_ref.extractall(f'{output}')
    shutil.move(f'{output}/xwave-cf862c1dc3ead168db68abe4c870a509285fc729',
                f'{output}/MSCMinimal')
    os.remove(zip_target)
    return f'{output}/MSCMinimal'


def minimal_msc_all_pointers(dir, sub=None, task=None, ses=None):
    if sub is None:
        sub = '*'
    if ses is None:
        ses = '*'
    if task is None:
        task = '*'
    paths = pathlib.Path(f'{dir}/data/MSC/ts/').glob(
        f'sub-MSC{sub}_ses-func{ses}_task-{task}_*ts.1D')
    paths = list(paths)
    paths.sort()
    paths = [(
        str(path),
        f"{re.sub(r'/ts/', '/motion/', str(path))[:-17]}relRMS.1D"
    ) for path in paths]
    return paths


class MSCMinimal:
    """
    Minimal version of the Midnight Scan Club dataset.
    """
    #TODO: add option to split by subject or session
    #TODO: gotta implement DataLoader / torch data compatibility for
    # parallelisation. or use async.
    def __init__(self, delete_if_exists=False, #all_in_memory=False,
                 shuffle=True, batch_size=1, sub=None, task=None, ses=None,
                 dtype=torch.float, device=None, rms_thresh=None):
        self.srcdir = minimal_msc_download(delete_if_exists=delete_if_exists)
        self.paths = minimal_msc_all_pointers(
            self.srcdir, sub=sub, task=task, ses=ses)
        self.max = len(self.paths)
        self.shuffle = shuffle
        self._cfg_iter(shuffle=shuffle)
        self.to_tensor = ToTensor(dtype=dtype, device=device)
        self.rms_thresh = rms_thresh
        self.batch_size = batch_size
        self.idx = -self.batch_size

    def _load_single(self, idx):
        # As of now, we force all proposed indices into the valid range
        if idx > self.max:
            idx = random.randint(0, self.max - 1)
        bold, motion = self.paths[idx]
        bold = pd.read_csv(bold, sep=' ', header=None)
        ret = {
            '__id__': self.paths[idx][0].split('/')[-1][:-18],
            'bold': self.to_tensor(bold)
        }
        if self.rms_thresh is not None:
            motion = pd.read_csv(motion, sep=' ', header=None)
            tmask = (motion <= self.rms_thresh)
            ret.update({
                'tmask': to_tensor(tmask, dtype=torch.bool,
                                   device=self.to_tensor.device)
            })
        return ret

    def _load_batch(self):
        start = self.idx
        end = start + self.batch_size
        batch_list = [
            self._load_single(self.idxmap[idx])
            for idx in range(start, end)
        ]
        batch = {}
        batch['__id__'] = [e['__id__'] for e in batch_list]
        batch['bold'] = torch.stack(
            extend_to_max_size(
                [e['bold'] for e in batch_list],
                fill=0),
        0)
        batch['tmask'] = torch.stack(
            extend_to_max_size(
                [e['tmask'] for e in batch_list],
                fill=False),
        0)
        return batch

    def _cfg_iter(self, shuffle):
        idxmap = list(range(self.max))
        if shuffle:
            random.shuffle(idxmap)
        self.idxmap = idxmap

    def __iter__(self):
        return self

    def __next__(self):
        self.idx += self.batch_size
        if self.idx < self.max:
            return self._load_batch()
        self._cfg_iter(shuffle=self.shuffle)
        raise StopIteration
