# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Neuroimaging data search
~~~~~~~~~~~~~~~~~~~~~~~~
A lightweight pybids alternative for fMRIPrep-like processed data that isn't as
particular about leading zeros. It's not very robust and is likely to be a
temporary solution until pybids is stabilised.
"""
import re, pathlib


regexs = {
    'datatype': '.*/(?P<datatype>[^/]*)/[^/]*',
    'subject': '.*/[^/]*sub-(?P<subject>[^_]*)[^/]*',
    'session': '.*/[^/]*ses-(?P<subject>[^_]*)[^/]*',
    'run': '.*/[^/]*run-(?P<run>[^_]*)[^/]*',
    'task': '.*/[^/]*task-(?P<task>[^_]*)[^/]*',
    'space': '.*/[^/]*space-(?P<space>[^_]*)[^/]*',
    'desc': '.*/[^/]*desc-(?P<desc>[^_]*)[^/]*',
    'suffix': '.*/[^/]*_(?P<suffix>[^/_\.]*)\..*',
    'extension': '.*/[^/\.]*(?P<extension>\..*)$'
}


class LightBIDSObject(object):
    def __init__(self, path):
        self.pathobj = path
        self.path = str(path)
        self.parse_path()

    def __repr__(self):
        return (
            f'LightBIDSObject({self.pathobj.name}, '
            f'dtype={self.datatype})'
        )

    def parse_path(self):
        vals = {k: re.match(v, self.path) for k, v in regexs.items()}
        vals = {k: v.groupdict()[k] for k, v in vals.items() if v is not None}
        for k, v in vals.items():
            self.__dict__[k] = v

    def get(self, key):
        return self.__dict__.get(key)


class LightGrabber(object):
    def __init__(self, root, patterns=None):
        self.refs = []
        self.root = root
        patterns = patterns or ['*']
        for pattern in patterns:
            files = self.find_files(pattern)
            for f in files:
                self.refs += [LightBIDSObject(f)]

    def find_files(self, pattern):
        return pathlib.Path(self.root).glob(f'**/{pattern}')

    def get_subjects(self):
        try:
            out = list(set([r.subject for r in lg.refs]))
            out.sort()
            return out
        except AttributeError:
            return None

    def get_sessions(self):
        try:
            out = list(set([r.session for r in lg.refs]))
            out.sort()
            return out
        except AttributeError:
            return []

    def get_runs(self):
        try:
            out = list(set([r.run for r in lg.refs]))
            out.sort()
            return out
        except AttributeError:
            return []

    def get_tasks(self):
        try:
            out = list(set([r.task for r in lg.refs]))
            out.sort()
            return out
        except AttributeError:
            return []

    def get(self, **filters):
        obj = self.refs
        for k, v in filters.items():
            if k not in regexs.keys():
                continue
            obj = [r for r in obj if r.get(k) == v]
        return obj
