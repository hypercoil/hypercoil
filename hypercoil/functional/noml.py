# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Null-option multi-logit
~~~~~~~~~~~~~~~~~~~~~~~
Softmax domain mapper with a null option. Needs its own submodule so we don't
get a circular import.
"""
import torch
from .domain import _Domain, _PhaseAmplitudeDomain
from ..init.base import ConstantInitialiser


class NullOptionMultiLogit(_Domain):
    def __init__(self, axis=-1, handler=None, minim=1e-3, null_init=None):
        super(NullOptionMultiLogit, self).__init__(
            handler=handler, bound=(minim, 1 - minim))
        self.axis = axis
        self.null_init = null_init or ConstantInitialiser(0)
        self.signature[axis] = (
            lambda x: x + 1,
            lambda x: x - 1
        )

        def preimage_map(x):
            dim = list(x.size())
            dim[self.axis] = 1
            nulls = torch.empty(dim)
            self.null_init(nulls)
            nulls = self.handler.apply(nulls, self.bound)
            z = torch.cat((x, nulls), self.axis)
            return torch.log(z)

        def image_map(x):
            x = torch.softmax(x, self.axis)
            return x.index_select(
                self.axis,
                torch.arange(0, x.size(self.axis) - 1)
            )

        self.preimage_map = preimage_map
        self.image_map = image_map


class ANOML(_PhaseAmplitudeDomain, NullOptionMultiLogit):
    pass
