# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Regularisation scheme
~~~~~~~~~~~~~~~~~~~~~
Module for additively applying a set of several regularisations to a tensor.
"""
import torch
from torch.nn import Module


class RegularisationScheme(Module):
    def __init__(self, reg=None):
        super(RegularisationScheme, self).__init__()
        self.reg = self._listify(reg) or []

    def __add__(self, other):
        return RegularisationScheme(reg=(self.reg + other.reg))

    def __iadd__(self, other):
        self.reg += other
        return self

    def __iter__(self):
        self.n = 0
        return self

    def __len__(self):
        return len(self.reg)

    def __next__(self):
        if self.n < len(self.reg):
            self.n += 1
            return self.reg[self.n - 1]
        else:
            raise StopIteration

    def __repr__(self):
        s = [f'\n    {r}' for r in self.reg]
        s = ','.join(s)
        return f'RegularisationScheme({s}\n)'

    def __getitem__(self, key):
        return self.reg[key]

    def _listify(self, x):
        if x is None:
            return None
        if not isinstance(x, list):
            return list(x)
        return x

    def forward(self, weight):
        losses = torch.tensor(0., requires_grad=True)
        for r in self:
            losses = losses + r(weight)
        return losses
