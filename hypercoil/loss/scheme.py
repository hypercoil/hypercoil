# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Module for additively applying a set of several losses or
regularisations to a set of inputs.
"""
import torch
from .base import LossApply, UnpackingLossArgument
from ..engine import SentryModule
from ..engine.terminal import Terminal


def identity(*args):
    if len(args) == 1:
        return args[0]
    return args


class LossScheme(SentryModule):
    def __init__(self, loss=None, apply=None):
        super(LossScheme, self).__init__()
        self.loss = self._listify(loss) or []
        if apply is None:
            apply = identity
        self.apply = apply
        self.name = 'Scheme'

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

    def register_sentry(self, sentry):
        for f in self:
            ##TODO: change when we implement sentry functionality for Terminal
            if not isinstance(f, Terminal):
                f.register_sentry(sentry)

    def register_action(self, action):
        for f in self:
            ##TODO: change when we implement sentry functionality for Terminal
            if not isinstance(f, Terminal):
                f.register_action(action)

    def forward(self, *args, verbose=True, **kwargs):
        losses = 0
        applied = self.apply(*args, **kwargs)
        if verbose:
            for f in self:
                if isinstance(f, LossScheme):
                    if isinstance(applied, UnpackingLossArgument):
                        loss = f(**applied, verbose=True)
                    else:
                        loss = f(applied, verbose=True)
                elif (isinstance(f, LossApply) and
                    isinstance(f.loss, LossScheme)):
                    if isinstance(applied, UnpackingLossArgument):
                        loss = f.loss(
                            f.apply(**applied), verbose=True)
                    else:
                        loss = f.loss(
                            f.apply(applied), verbose=True)
                else:
                    loss = f(applied)
                    print(f'- {f}: {loss}')
                # Terminals will automatically send gradients back along
                # their lines. We should not duplicate this if a terminal is
                # nested.
                if isinstance(f, Terminal):
                    continue
                elif (isinstance(f, LossApply) and
                    isinstance(f.loss, Terminal)):
                    continue
                losses = losses + loss
        else:
            for f in self:
                if isinstance(applied, UnpackingLossArgument):
                    loss = f(**applied)
                else:
                    loss = f(applied)
                # Terminals will automatically send gradients back along
                # their lines. We should not duplicate this if a terminal is
                # nested.
                if isinstance(f, Terminal):
                    continue
                elif (isinstance(f, LossApply) and
                    isinstance(f.loss, Terminal)):
                    continue
                losses = losses + loss
        return losses
