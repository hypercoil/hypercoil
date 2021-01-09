# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Neuroimaging
~~~~~~~~~~~~
Neuroimaging dataset-specific datasets and data references.
"""
from .dataref import DataReference
from .transforms import (
    ReadNiftiTensorBlock,
    ReadTableTensorBlock,
    ToTensor, EncodeOneHot
)


class fMRISubReference(DataReference):
    def __init__(
        self,
        data,
        confounds=None,
        labels=None,
        outcomes = None,
        data_transform=None,
        confounds_transform=None,
        label_transform=None,
        **ids):
        self.data_ref = data
        self.data_transform = data_transform or ReadNiftiTensorBlock()
        self.confounds_ref = confounds
        self.confounds_transform = (
            confounds_transform or ReadTableTensorBlock())
        self.label_ref = labels or []
        self.outcome_ref = outcomes or []
        self.label_transform = label_transform or IdentityTransform()
        self.subject = ids.get('subject')
        self.session = ids.get('session')
        self.run = ids.get('run')
        self.task = ids.get('task')

    def __repr__(self):
        s = f'{type(self).__name__}(sub={self.subject}'
        if self.session:
            s += f', ses={self.session}'
        if self.run:
            s += f', run={self.run}'
        if self.task:
            s += f', task={self.task}'
        s += ')'
        return s


class fMRIDataReference(DataReference):
    def __init__(self, df, idx, level_names=None,
                 variables=None, labels=None, outcomes=None):
        super(fMRIDataReference, self).__init__(
            data=df, idx=idx, level_names=level_names,
            variables=variables, labels=labels, outcomes=outcomes)

        self.subrefs = self.make_subreferences()

        self.subject = self.ids.get('subject')
        self.session = self.ids.get('session')
        self.run = self.ids.get('run')
        self.task = self.ids.get('task')

    def make_subreferences(self):
        subrefs = []
        for idx, ref in self.df.iterrows():
            ids = dict(zip(self.df.index.names, idx))
            subrefs += [fMRISubReference(
                data=ref.images,
                confounds=ref.confounds,
                labels=self.labels,
                outcomes=self.outcomes,
                data_transform=self.variables[0].transform,
                confounds_transform=self.variables[1].transform,
                label_transform=self.labels[0].transform,
                **ids
            )]
        return subrefs

    def __repr__(self):
        s = f'{type(self).__name__}(sub={self.subject}'
        if self.session:
            s += f', ses={self.session}'
        if self.run:
            s += f', run={self.run}'
        if self.task:
            s += f', task={self.task}'
        for sr in self.subrefs:
            s += f',\n {sr}'
        s += '\n)'
        return s
