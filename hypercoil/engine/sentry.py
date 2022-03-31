# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Sentry
~~~~~~
Elementary sentry objects and actions.
"""
import torch
from torch.nn import Module
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import partial


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


class PropagateMultiplierFromTransform(SentryAction):
    def __init__(self, transform):
        super().__init__(trigger=['EPOCH'])
        self.transform = transform


class PropagateMultiplierFromEpochTransform(
    PropagateMultiplierFromTransform
):
    def propagate(self, sentry, received):
        message = {'NU': self.transform(received['EPOCH'])}
        for s in sentry.listeners:
            s._listen(message)


class PropagateMultiplierFromRecursiveTransform(
    PropagateMultiplierFromTransform
):
    def propagate(self, sentry, received):
        for s in sentry.listeners:
            message = {'NU': self.transform(s.nu)}
            s._listen(message)


class UpdateMultiplier(SentryAction):
    def __init__(self):
        super().__init__(trigger=['NU'])

    def propagate(self, sentry, received):
        sentry.nu = received['NU']


class ArchiveLoss(SentryAction):
    def __init__(self):
        super().__init__(trigger=['LOSS', 'NAME', 'NU'])

    def propagate(self, sentry, received):
        name = received['NAME']
        sentry.archive[name] += [received['LOSS']]
        sentry.archive[f'{name}_nu'] += [received['NU']]


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


class SchedulerSentry(Sentry):
    def __init__(self, epochs, base=1):
        super().__init__()
        self.base = base
        epochs.register_sentry(self)


class MultiplierSchedule(SchedulerSentry):
    def __init__(self, epochs, transform, base=1):
        super().__init__(epochs=epochs, base=base)
        self.register_action(
            PropagateMultiplierFromEpochTransform(transform=transform))


class MultiplierRecursiveSchedule(SchedulerSentry):
    def __init__(self, epochs, transform, base=1):
        super().__init__(epochs=epochs, base=base)
        self.register_action(
            PropagateMultiplierFromRecursiveTransform(transform=transform))


class MultiplierSigmoidSchedule(MultiplierSchedule):
    def __init__(self, epochs, transitions, base=1):
        cur = base
        for k, v in transitions.items():
            transitions[k] = (cur, v)
            cur = v
        transform = partial(MultiplierSigmoidSchedule.get_transform,
                            transitions=transitions)
        super().__init__(
            epochs=epochs,
            transform=transform,
            base=base
        )

    @staticmethod
    def get_transform(e, transitions):
        for ((begin_epoch, end_epoch),
             (begin_nu, end_nu)) in transitions.items():
            if e < begin_epoch:
                return begin_nu
            elif e < end_epoch:
                x_scale = (end_epoch - begin_epoch)
                y_scale = (end_nu - begin_nu)
                x = torch.tensor(e - (begin_epoch + x_scale / 2) + 0.5)
                y = begin_nu
                return y_scale * torch.sigmoid(x).item() + y
        return end_nu


class LossArchive(Sentry):
    def __init__(self):
        super().__init__()
        self.archive = {}
        self.register_action(ArchiveLoss())

    def _register_trigger(self, sentry):
        self.archive[sentry.name] = []
        self.archive[f'{sentry.name}_nu'] = []

    def get(self, name, normalised=False):
        tape = self.archive[name]
        if normalised:
            nu_tape = self.archive[f'{name}_nu']
            return [loss / nu for (loss, nu) in zip(tape, nu_tape)]
        return tape
