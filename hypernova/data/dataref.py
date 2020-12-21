# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data references
~~~~~~~~~~~~~~~
Interfaces between neuroimaging data and data loaders.
"""
from .transforms import ReadNiftiTensor, ReadTableTensor, IdentityTransform


class DataReference(object):
    """Does nothing. Yet."""
    def __init__(self):
        pass


class FunctionalConnectivityDataReference(DataReference):
    def __init__(
        self,
        data,
        data_transform=None,
        confounds=None,
        confounds_transform=None,
        label=None,
        label_transform=None,
        **ids):
        self.data_ref = data
        self.data_transform = data_transform or ReadNiftiTensor()
        self.confounds_ref = confounds
        self.confounds_transform = confounds_transform or ReadTableTensor()
        self.label_ref = label or 'sub'
        self.label_transform = label_transform or IdentityTransform()
        self.subject = ids.get('subject')
        self.session = ids.get('session')
        self.run = ids.get('run')
        self.task = ids.get('task')

    @property
    def data(self):
        return self.data_transform(self.data_ref)

    @property
    def confounds(self):
        return self.confounds_transform(self.confounds_ref)

    @property
    def label(self):
        return self.label_transform(self.label_ref)

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
