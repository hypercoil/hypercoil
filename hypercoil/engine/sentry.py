# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Sentry
~~~~~~
Elementary sentry objects and actions.
"""
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
        if self not in sentry.listening:
            sentry.listening += [self]

    def register_action(self, action):
        if action not in self.actions:
            self.actions += [action]
        if self not in action.sentries:
            action.sentries += [self]

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
        received = message.get(self.trigger)
        if received:
            for sentry in self.sentries:
                self.propagate(sentry, received)

    def register(self, sentry):
        if self not in sentry.actions:
            sentry.actions += [self]
        if sentry not in self.sentries:
            self.sentries += [sentry]

    @abstractmethod
    def propagate(self, received):
        pass


class PropagateMultiplierFromEpochTransform(SentryAction):
    def __init__(self, transform):
        super().__init__(trigger='EPOCH')
        self.transform = transform

    def propagate(self, sentry, received):
        message = {'NU': self.transform(received)}
        for s in sentry.listeners:
            s._listen(message)


class UpdateMultiplier(SentryAction):
    def __init__(self):
        super().__init__(trigger='NU')

    def propagate(self, sentry, received):
        sentry.nu = received


class ArchiveLoss(SentryAction):
    def __init__(self):
        super().__init__(trigger='LOSS')

    def propagate(self, sentry, received):
        sentry.archive += [received]


class Epochs(Sentry, Iterator):
    def __init__(self, max_epoch):
        self.cur_epoch = -1
        self.max_epoch = max_epoch
        super().__init__()

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


class MultiplierSchedule(Sentry):
    def __init__(self, epochs, transform, base=1):
        super().__init__()
        self.base = base
        self.register_action(
            PropagateMultiplierFromEpochTransform(transform=transform))
        epochs.register_sentry(self)


class LossArchive(Sentry):
    def __init__(self):
        super().__init__()
        self.archive = []
        self.register_action(ArchiveLoss())
