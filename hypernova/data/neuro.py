# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Neuroimaging
~~~~~~~~~~~~
Neuroimaging dataset-specific datasets and data references.
"""
from .dataref import DataReference


class fMRIDataReference(DataReference):
    def __init__(self, df, idx, level_names=None,
                 variables=None, labels=None, outcomes=None):
        super(fMRIDataReference, self).__init__(
            data=df, idx=idx, level_names=level_names,
            variables=variables, labels=labels, outcomes=outcomes)

        if self.level_names:
            self.subrefs = self.make_subreferences()
        else:
            self.subrefs = []

        self.subject = self.ids.get('subject')
        self.session = self.ids.get('session')
        self.run = self.ids.get('run')
        self.task = self.ids.get('task')

    def make_subreferences(self):
        subrefs = []
        for idx, _ in self.df.iterrows():
            ids = dict(zip(self.df.index.names, idx))
            subrefs += [fMRIDataReference(
                df=self.df,
                idx=idx,
                level_names=None,
                variables=self.vfactory,
                labels=self.lfactory,
                outcomes=self.ofactory
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
        if not self.subrefs:
            s += ')'
            return s
        for sr in self.subrefs:
            s += f',\n {sr}'
        s += '\n)'
        return s
