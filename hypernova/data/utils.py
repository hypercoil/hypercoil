# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities
~~~~~~~~~
Utility functions for loading and manipulating data before they are ingested
by a model.
"""
import re, json
import numpy as np


def load_metadata(path):
    with open(path) as file:
        metadata = json.load(file)
    return metadata


def diff_nanpad(a, n=1, axis=-1):
    diff = np.empty_like(a) * float('nan')
    diff[n:] = np.diff(a, n=n, axis=axis)
    return diff


def numbered_string(s):
    num = int(re.search('(?P<num>[0-9]+$)', s).groupdict()['num'])
    string = re.sub('[0-9]+$', '', s)
    return (string, num)
