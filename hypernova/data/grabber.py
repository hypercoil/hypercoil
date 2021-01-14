# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Neuroimaging data search
~~~~~~~~~~~~~~~~~~~~~~~~
A lightweight pybids alternative. Not robust; a temporary solution.
"""
import re, pathlib


class LightGrabber(object):
    """
    Extremely lightweight grabbit-like system for representation of data
    organised in a structured way across multiple files.

    A lightweight pybids alternative for fMRIPrep-like processed data that
    isn't as particular about leading zeros. It's not very robust and is
    likely to be a temporary solution until pybids is stabilised.

    Parameters
    ----------
    root : str
        Path to the root directory of the dataset on the file system.
    template : VariableFactory
        Factory for producing variables to assign each path. Consider using a
        factory for a subclass of `DataPathVariable` if the data instances
        have associated metadata.
    patterns : list(str)
        String patterns to constrain the scope of the layout. If patterns are
        provided, then the layout will not include any files that do not match
        at least one pattern.
    queries : list(DataQuery)
        Query objects defining the variables to extract from the dataset via
        query.

    Attributes
    ----------
    refs : List(DatasetVariable)
        List of the variables produced by the factory provided as `template`.
    """
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
        """
        Find files in the dataset directory (and any nested directories) that
        match a specified pattern.
        """
        return pathlib.Path(self.root).glob(f'**/{pattern}')

    def getall(self, entity):
        """
        Find all values that a particular data entity (e.g., an identifier)
        takes in the dataset files included in the layout.

        Parameters
        ----------
        entity : str
            Name of the entity (e.g., 'subject' or 'run') to enumerate.

        Returns
        -------
        list
            List of all unique values that the specified entity assumes in the
            dataset.
        """
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
        """
        Find dataset objects that match the specified filters.
        """
        obj = self.refs
        for k, v in filters.items():
            if (self.attr_regex) and (k not in self.attr_regex.keys()):
                continue
            obj = [r for r in obj if r.get(k) == v]
        return obj
