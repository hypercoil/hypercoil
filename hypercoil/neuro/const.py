# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Constants
~~~~~~~~~
Constants for import.
"""
from neuromaps.datasets import fetch_fsaverage


def neuromaps_fetch_fn(
    nmaps_fetch_fn: callable,
    density: str,
    suffix: str,
    hemi: int,
):
    hemi_map = {
        "L": 0,
        "R": 1,
    }
    return nmaps_fetch_fn(density=density)[suffix][hemi_map[hemi]]


def template_dict():
    return {
        "fsLR": fsLR(),
        "fsaverage": fsaverage(),
    }


class fsaverage:
    NMAPS_MASK_QUERY = {
        "nmaps_fetch_fn": fetch_fsaverage,
        "density": "41k",
        "suffix": "medial",
    }
    NMAPS_COOR_QUERY = {
        "nmaps_fetch_fn": fetch_fsaverage,
        "density": "41k",
        "suffix": "sphere",
    }
    NMAPS_COMPARTMENTS = {
        "L": 0,
        "R": 1,
    }
    TFLOW_COOR_QUERY = {
        "template": "fsLR",
        "space": None,
        "density": "32k",
        "suffix": "sphere",
    }
    TFLOW_COMPARTMENTS = {
        "L": {"hemi": "lh"},
        "R": {"hemi": "rh"},
    }


class fsLR:
    TFLOW_MASK_QUERY = {
        "template": "fsLR",
        "density": "32k",
        "desc": "nomedialwall",
    }
    TFLOW_COOR_QUERY = {
        "template": "fsLR",
        "space": None,
        "density": "32k",
        "suffix": "sphere",
    }
    TFLOW_COMPARTMENTS = {
        "L": {"hemi": "L"},
        "R": {"hemi": "R"},
    }


class CIfTIStructures:
    LEFT = "CIFTI_STRUCTURE_CORTEX_LEFT"
    RIGHT = "CIFTI_STRUCTURE_CORTEX_RIGHT"
