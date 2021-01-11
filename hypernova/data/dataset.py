# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Referenced datasets
~~~~~~~~~~~~~~~~~~~
Dataset objects composed of DataReference subclasses.
"""
from itertools import chain
from torch.utils.data import Dataset, DataLoader


class ReferencedDataset(Dataset):
    def __init__(self, data_refs, depth=0):
        self.depth = depth
        self._data_refs = data_refs
        self.data_refs = self._refs_at_depth()

    def _refs_at_depth(self):
        refs = self._data_refs
        depth = 0
        while depth < self.depth:
            try:
                refs = [r.subrefs for r in refs]
                refs = list(chain(*refs))
            except AttributeError:
                raise AttributeError(
                    f'Proposed reference depth too great: {self.depth}')
            depth += 1
        return refs

    def set_depth(self, depth):
        cur_depth = self.depth
        self.depth = depth
        try:
            self.data_refs = self._refs_at_depth()
        except AttributeError:
            self.depth = cur_depth
            raise

    def add_data(self, data_refs):
        self._data_refs += data_refs
        self.data_refs = self._refs_at_depth()

    def __len__(self):
        return len(self.data_refs)

    def __getitem__(self, idx):
        ref = self.data_refs[idx]
        return ref()

    def __repr__(self):
        return f'{type(self).__name__}(n={len(self)}, depth={self.depth})'
