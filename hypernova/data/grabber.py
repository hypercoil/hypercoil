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


class LightGrabber(object):
    def __init__(self, root, template, patterns=None, queries=None):
        self.refs = []
        self.root = root
        try:
            self.attr_regex = template.regex
        except AttributeError:
            self.attr_regex = {}
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
                f_var = template(name=f.name)
                f_var.assign(f)
                self.refs += [f_var]

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
            if (self.attr_regex) and (k not in self.attr_regex.keys()):
                continue
            obj = [r for r in obj if r.get(k) == v]
        return obj
