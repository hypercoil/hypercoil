# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Sentry
~~~~~~
Elementary sentry objects and actions.
"""
import torch
import pandas as pd
from torch.nn import Module
from abc import ABC, abstractmethod
from collections.abc import Iterator


class Sentry:
    def __init__(self):
        self.listeners = []
        self.listening = []
        self.actions = []

    def register_sentry(self, sentry):
        if sentry not in self.listeners:
            self.listeners += [sentry]
            self._register_sentry_extra(sentry)
        if self not in sentry.listening:
            sentry.listening += [self]
            sentry._register_trigger(self)

    def register_action(self, action):
        if action not in self.actions:
            self.actions += [action]
            self._register_action_extra(action)
        if self not in action.sentries:
            action.sentries += [self]
            action._register_trigger(self)

    def _register_sentry_extra(self, sentry):
        pass

    def _register_action_extra(self, sentry):
        pass

    def _register_trigger(self, sentry):
        pass

    def _listen(self, message):
        for action in self.actions:
            action(message)


class SentryModule(Module, Sentry):
    def __init__(self):
        super().__init__()
        self.listeners = []
        self.listening = []
        self.actions = []


class SentryAction(ABC):
    def __init__(self, trigger):
        self.trigger = trigger
        self.sentries = []

    def __call__(self, message):
        received = {t: message.get(t) for t in self.trigger}
        if None not in received.values():
            for sentry in self.sentries:
                self.propagate(sentry, received)

    def register(self, sentry):
        if self not in sentry.actions:
            sentry.actions += [self]
        if sentry not in self.sentries:
            self.sentries += [sentry]

    def _register_trigger(self, sentry):
        pass

    @abstractmethod
    def propagate(self, received):
        pass


class Epochs(Sentry, Iterator):
    def __init__(self, max_epoch):
        self.cur_epoch = -1
        self.max_epoch = max_epoch
        super().__init__()

    def reset(self):
        self.cur_epoch = -1

    def set(self, epoch):
        self.cur_epoch = epoch

    def set_max(self, epoch):
        self.max_epoch = epoch

    def __iter__(self):
        return self

    def __next__(self):
        self.cur_epoch += 1
        if self.cur_epoch >= self.max_epoch:
            raise StopIteration
        message = {'EPOCH' : self.cur_epoch}
        for s in self.listeners:
            s._listen(message)
        return self.cur_epoch
