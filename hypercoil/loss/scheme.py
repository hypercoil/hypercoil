# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Loss scheme
~~~~~~~~~~~
Module for additively applying a set of several losses or
regularisations to a set of inputs.
"""
import torch
from torch.nn import Module
from .base import LossApply


def identity(*args):
    if len(args) == 1:
        return args[0]
    return args


class LossScheme(Module):
    def __init__(self, loss=None, apply=None):
        super(LossScheme, self).__init__()
        self.loss = self._listify(loss) or []
        if apply is None:
            apply = identity
        self.apply = apply

    def __add__(self, other):
        return LossScheme(loss=(self.loss + other.loss))

    def __iadd__(self, other):
        self.loss += other
        return self

    def __iter__(self):
        self.n = 0
        return self

    def __len__(self):
        return len(self.loss)

    def __next__(self):
        if self.n < len(self.loss):
            self.n += 1
            return self.loss[self.n - 1]
        else:
            raise StopIteration

    def __repr__(self):
        s = [f'\n    {r}' for r in self.loss]
        s = ','.join(s)
        return f'LossScheme({s}\n)'

    def __getitem__(self, key):
        return self.loss[key]

    def _listify(self, x):
        if x is None:
            return None
        if not isinstance(x, list):
            return list(x)
        return x

    def forward(self, *args, verbose=False, **kwargs):
        losses = 0
        if verbose:
            for f in self:
                if isinstance(f, LossScheme):
                    loss = f(self.apply(*args, **kwargs), verbose=True)
                elif (isinstance(f, LossApply) and
                    isinstance(f.loss, LossScheme)):
                    loss = f.loss(
                        f.apply(self.apply(*args, **kwargs)),
                        verbose=True
                    )
                else:
                    loss = f(self.apply(*args, **kwargs))
                    print(f'- {f}: {loss}')
                losses = losses + loss
        else:
            for f in self:
                losses = losses + f(self.apply(*args, **kwargs))
        return losses
