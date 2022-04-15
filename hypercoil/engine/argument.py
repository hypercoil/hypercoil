# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Model argument
~~~~~~~~~~~~~~
Convenience mappings representing arguments to a model or loss function.
Also aliased to loss arguments.
"""
from collections.abc import Mapping


class ModelArgument(Mapping):
    """
    Effectively this is currently little more than a prettified dict.
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def __setitem__(self, k, v):
        self.__setattr__(k, v)

    def __getitem__(self, k):
        return self.__dict__[k]

    def __delitem__(self, k):
        del self.__dict__[k]

    def __len__(self):
        return len(self.__dict__)

    def __iter__(self):
        return iter(self.__dict__)


class UnpackingModelArgument(ModelArgument):
    """
    ``ModelArgument`` variant that is automatically unpacked when it is the
    output of an ``apply`` call.
    """
    pass
