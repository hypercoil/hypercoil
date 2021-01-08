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


class fMRIReferenceBase(DataReference):
    @property
    def data(self):
        return self.data_ref()

    @property
    def confounds(self):
        return self.confounds_ref()

    @property
    def label(self):
        return {k: l.transform(v) for l, (k, v) in
                zip(self.labels, self.label_ref.items())}

    @property
    def outcome(self):
        return {k: o.transform(v) for o, (k, v) in
                zip(self.outcomes, self.outcome_ref.items())}


class fMRISubReference(fMRIReferenceBase):
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


class fMRIDataReference(fMRIReferenceBase):
    def __init__(self, df, idx,
                 variables=None, labels=None, outcomes=None,
                 data_transform=None, confounds_transform=None,
                 label_transform=None, outcome_transform=None,
                 level_names=None):
        super(fMRIDataReference, self).__init__(
            data=df, idx=idx, level_names=level_names,
            variables=variables, labels=labels, outcomes=outcomes)
        self.df = df.loc(axis=0)[idx]

        self.data_ref = self.variables[0]
        self.confounds_ref = self.variables[1]
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
