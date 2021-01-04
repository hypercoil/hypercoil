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
            try:
                self.__dict__[k] = int(v)
            except ValueError:
                self.__dict__[k] = v

    def get(self, key):
        try:
            return self.__dict__.get(int(key))
        except ValueError:
            return self.__dict__.get(key)


class LightGrabber(object):
    def __init__(self, root, template, patterns=None, queries=None):
        self.refs = []
        self.root = root
        if queries:
            qpatterns = [q.pattern for q in queries if q.pattern is not None]
            if patterns:
                patterns += qpatterns
            else:
                patterns = qpatterns
        patterns = patterns or ['*']
        for pattern in patterns:
            files = self.find_files(pattern)
            for f in files:
                self.refs += [template(f)]

    def find_files(self, pattern):
        return pathlib.Path(self.root).glob(f'**/{pattern}')

    def getall(self, entity):
        try:
            out = list(set([r.get(entity) for r in self.refs]))
            try:
                out.sort()
            except TypeError:
                pass
            if len(out) == 1 and out[0] is None:
                return []
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


class LightBIDSLayout(LightGrabber):
    def __init__(self, root, patterns=None, queries=None):
        super(LightBIDSLayout, self).__init__(
            root=root,
            patterns=patterns,
            queries=queries,
            template=LightBIDSObject
        )
