# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Documentation utilities.
"""
from __future__ import annotations
from collections import UserDict


class NestedDocParse(UserDict):
    """
    Enable multiple documentation decorators to be applied to a single
    function, with each pass leaving intact any cells that are not specified
    in the current decorator.
    """

    def __missing__(self, key):
        return f"{{{key}}}"
