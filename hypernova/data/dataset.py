# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Neuroimaging datasets
~~~~~~~~~~~~~~~~~~~~~
Dataset objects for neuroimaging data.
"""
import nibabel as nb
from torch.utils.data import Dataset, DataLoader


class NeuroimagingDataset(Dataset):
    def __init__(self, data_refs, labels='sub'):
        self.data_refs = data_refs

    def __len__(self):
        return len(self.data_refs)

    def __getitem__(self, idx):
        ref = self.data_refs[idx]
        return {
            'data': ref.data,
            'confounds': ref.confounds,
            'label': ref.label
        }


class fMRIPrepDataset(NeuroimagingDataset):
    def __init__(self, fmriprep_dir):
        data_refs = fmriprep_references(fmriprep_dir)
        super(fMRIPrepDataset, self).__init__(data_refs)
