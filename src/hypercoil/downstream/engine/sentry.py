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
from collections.abc import Iterator, Mapping


class Sentry:
    def __init__(self):
        self.listeners = []
        self.listening = []
        self.actions = []
        self.message = SentryMessage()

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
        self.message = SentryMessage()


class SentryMessage(Mapping):
    def __init__(self, **kwargs):
        super().__init__()
        self.content = {}
        self.content.update(kwargs)

    def __setitem__(self, k, v):
        self.__setattr__(k, v)

    def __getitem__(self, k):
        return self.content[k]

    def __delitem__(self, k):
        del self.content[k]

    def __len__(self):
        return len(self.content)

    def __iter__(self):
        return iter(self.content)

    def _fmt_tsr_repr(self, tsr):
        return f'<tensor of dimension {tuple(tsr.shape)}>'

    def __repr__(self):
        s = f'{type(self).__name__}('
        for k, v in self.items():
            if isinstance(v, torch.Tensor):
                v = self._fmt_tsr_repr(v)
            elif isinstance(v, list):
                v = f'<list with {len(v)} elements>'
            elif isinstance(v, tuple):
                fmt = [self._fmt_tsr_repr(i)
                       if isinstance(i, torch.Tensor)
                       else i for i in v]
                v = f'({fmt})'
            s += f'\n    {k} : {v}'
        s += ')'
        return s

    def clear(self):
        self.content = {}

    def reset(self):
        self.clear()

    def update(self, *args, **kwargs):
        for k, v in args:
            self.content[k] = v
        self.content.update(**kwargs)

    def transmit(self):
        return self.content


class SentryAction(ABC):
    def __init__(self, trigger):
        self.trigger = trigger
        self.sentries = []
        self.message = SentryMessage()

    def __call__(self, message):
        if self.trigger is None:
            for sentry in self.sentries:
                self.propagate(sentry, message)
            return
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
        self.message.update(('EPOCH', self.cur_epoch))
        for s in self.listeners:
            s._listen(self.message)
        return self.cur_epoch
