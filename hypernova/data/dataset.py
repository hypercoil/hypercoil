# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Referenced datasets
~~~~~~~~~~~~~~~~~~~
Dataset objects composed of DataReference subclasses.
"""
from itertools import chain
from functools import partial
from torch.utils.data import Dataset, DataLoader
from .collate import gen_collate, extend_and_bind


class ReferencedDataset(Dataset):
    """
    A referenced dataset.

    Parameters
    ----------
    data_refs : list(DataReference)
        List of data references to include in the dataset. References can be
        obtained using a search method like `data_references` or one of its
        parent functions.
    depth : int (default 0)
        Sampling depth of the dataset: if the passed references are nested
        hierarchically, then each minibatch is sampled from data references
        at the indicated level. By default, the dataset is sampled from the
        highest level (no nesting).
    """
    def __init__(self, data_refs, depth=0):
        self.depth = depth
        self._data_refs = data_refs
        self.data_refs = self._refs_at_depth()

    def _refs_at_depth(self):
        """
        Return the data references at the specified sampling depth.
        """
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
        """
        Adjust the dataset's sampling depth. If the dataset has nested
        references, then the depth corresponds to the level of nesting from
        which the data are sampled. References from the specified depth are
        automatically collated into the internal object list.
        """
        cur_depth = self.depth
        self.depth = depth
        try:
            self.data_refs = self._refs_at_depth()
        except AttributeError:
            self.depth = cur_depth
            raise

    def add_data(self, data_refs):
        """
        Append additional data references to the dataset.

        Parameters
        ----------
        data_refs : list(DataReference)
            List of additional references to add.
        """
        self._data_refs += data_refs
        self.data_refs = self._refs_at_depth()

    def __len__(self):
        return len(self.data_refs)

    def __getitem__(self, idx):
        ref = self.data_refs[idx]
        return ref()

    def __repr__(self):
        return f'{type(self).__name__}(n={len(self)}, depth={self.depth})'


class ReferencedDataLoader(DataLoader):
    """
    Data loader for a ReferencedDataset.
    """
    def __init__(self, dataset, **kwargs):
        kwargs=kwargs
        collate_fn = kwargs.get('collate_fn')
        if not collate_fn:
            kwargs['collate_fn'] = partial(
                gen_collate,
                concat=extend_and_bind,
                concat_axis=0
            )
        super(ReferencedDataLoader, self).__init__(dataset, **kwargs)

    def set_depth(self, depth):
        """
        Adjust the dataset's sampling depth. If the dataset has nested
        references, then the depth corresponds to the level of nesting from
        which the data are sampled.
        """
        self.dataset.set_depth(depth)
