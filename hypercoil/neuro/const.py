# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Constants
~~~~~~~~~
Constants for import.
"""


class fsLR:
    TFLOW_MASK_QUERY = {
        'template': 'fsLR',
        'density': '32k',
        'desc': 'nomedialwall'
    }
    TFLOW_COOR_QUERY = {
        'template': 'fsLR',
        'space': None,
        'density': '32k',
        'suffix': 'sphere'
    }
    TFLOW_COMPARTMENTS = {
        'L': {'hemi': 'L'},
        'R': {'hemi': 'R'}
    }