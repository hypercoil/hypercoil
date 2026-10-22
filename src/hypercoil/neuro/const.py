# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Constants for import.
"""
from typing import Optional

from lytemaps.datasets import fetch_fsaverage


def neuromaps_fetch_fn(
    nmaps_fetch_fn: callable,
    density: str,
    suffix: str,
    hemi: Optional[int] = None,
):
    """
    Fetch a surface template with a neuromaps-style fetcher.

    Parameters
    ----------
    nmaps_fetch_fn
        Fetcher, such as ``fetch_fsaverage``, that takes ``density`` and
        returns a mapping of template files.
    density
        Template density, such as ``'41k'``.
    suffix
        Key of the template in the mapping that the fetcher returns.
    hemi
        ``'L'`` or ``'R'`` selects one hemisphere. None returns both.

    Returns
    -------
    Any
        The template files of both hemispheres, or of ``hemi``.
    """
    template = nmaps_fetch_fn(density=density)[suffix]
    if hemi is not None:
        hemi_map = {
            'L': 0,
            'R': 1,
        }
        return template[hemi_map[hemi]]
    else:
        return template


def template_dict():
    """Return the template constants keyed by template name."""
    _fsLR = fsLR()
    _fsaverage = fsaverage()
    return {
        'fsLR': _fsLR,
        'fsaverage': _fsaverage,
        'fsaverage5': _fsaverage,
    }


class fsaverage:
    """Query constants of the fsaverage surface template."""
    NMAPS_MASK_QUERY = {
        'nmaps_fetch_fn': fetch_fsaverage,
        'density': '41k',
        'suffix': 'medial',
    }
    NMAPS_COOR_QUERY = {
        'nmaps_fetch_fn': fetch_fsaverage,
        'density': '41k',
        'suffix': 'sphere',
    }
    NMAPS_COMPARTMENTS = {
        'L': 0,
        'R': 1,
    }
    TFLOW_COOR_QUERY = {
        'template': 'fsLR',
        'space': None,
        'density': '32k',
        'suffix': 'sphere',
    }
    TFLOW_COMPARTMENTS = {
        'L': {'hemi': 'lh'},
        'R': {'hemi': 'rh'},
    }


class fsLR:
    """Query constants of the fsLR surface template."""
    TFLOW_MASK_QUERY = {
        'template': 'fsLR',
        'density': '32k',
        'desc': 'nomedialwall',
    }
    TFLOW_COOR_QUERY = {
        'template': 'fsLR',
        'space': None,
        'density': '32k',
        'suffix': 'sphere',
    }
    TFLOW_COMPARTMENTS = {
        'L': {'hemi': 'L'},
        'R': {'hemi': 'R'},
    }


class CIfTIStructures:
    """CIfTI structure names of the left and right cortex."""
    LEFT = 'CIFTI_STRUCTURE_CORTEX_LEFT'
    RIGHT = 'CIFTI_STRUCTURE_CORTEX_RIGHT'
