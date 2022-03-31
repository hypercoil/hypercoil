# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Sentry
~~~~~~
Generic sentry objects.
"""
from torch.nn import Module
from collections.abc import Iterator


class Sentry:
    def __init__(self):
        self.listeners = []
        self.listening = []

    def register_sentry(self, sentry):
        if sentry not in self.listeners:
            self.listeners += [sentry]
        if self not in sentry.listening:
            sentry.listening += [self]

    def _listen(self, message):
        for s in self.listeners:
            s._listen(message)


class SentryModule(Module):
    def __init__(self):
        super().__init__()
        self.listeners = []
        self.listening = []

    def register_sentry(self, sentry):
        if sentry not in self.listeners:
            self.listeners += [sentry]
        if self not in sentry.listening:
            sentry.listening += [self]

    def _listen(self, message):
        for s in self.listeners:
            s._listen(message)


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
    def __init__(self, epochs, loss, transform):
        super().__init__()
        self.loss = loss
        self.transform = transform
        epochs.register_sentry(self)

    def _listen(self, message):
        epoch = message.get('EPOCH')
        if epoch:
            self.loss.nu = self.transform(epoch)
            message = {'NU': self.loss.nu}
            for s in self.listeners:
                s._listen(message)
