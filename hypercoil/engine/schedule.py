# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Learning schedules
~~~~~~~~~~~~~~~~~~
Hyperparameter schedule representations for toggling and changing
hyperparameters as the program learns.
"""
import torch
from abc import ABC, abstractmethod


#TODO: If we ever end up revisiting this, there are a lot of really
# bad practices here.


thresh_condition = {
    '==': lambda v, t: v == t,
    '>=': lambda v, t: v >= t,
    '<=': lambda v, t: v <= t,
    '!=': lambda v, t: v != t,
    '>': lambda v, t: v > t,
    '<': lambda v, t: v < t
}


def check_thresh_condition(val, thr, condition):
    f = thresh_condition[condition]
    return f(val, thr)


class Trigger(ABC):
    def __init__(self, W=None):
        if W is not None:
            self.register(W)

    def register(self, W):
        self.W = W

    @abstractmethod
    def check(self):
        pass

    def __call__(self):
        return self.check()

    def __bool__(self):
        return self.check().item()


class EpochTrigger(Trigger):
    def __init__(self, epoch, W=None):
        super().__init__(W)
        self.epoch = epoch
        if self.epoch // 1 == self.epoch:
            self.format = 'absolute'
        else:
            self.format = 'relative'

    def check(self):
        if self.format == 'absolute':
            return self.W['epoch'].value >= self.epoch
        elif self.format == 'relative':
            return self.W['epoch'].value / self.W['epoch'].max_iter >= self.epoch


class LossTrigger(Trigger):
    def __init__(self, thr, condition='<', loss='loss', W=None):
        super().__init__(W)
        self.thr = thr
        self.condition = condition
        self.loss = loss

    def check(self):
        val = self.W[self.loss].value
        return check_thresh_condition(val, self.thr, self.condition)


class ConvergenceTrigger(Trigger):
    def __init__(self, thr, n_consecutive=1, loss='loss', W=None):
        super().__init__(W)
        self.thr = thr
        self.n_consecutive = n_consecutive
        self.loss = loss

    def check(self):
        val = self.W[self.loss].delta[-(self.n_consecutive + 1):]
        if val.size(0) < (self.n_consecutive + 1):
            return torch.tensor(False)
        return ((val[0] - val).abs().max() < self.thr)


class MultiTrigger(Trigger):
    def __init__(self, triggers, W=None):
        super().__init__()
        self.triggers = list(triggers)
        if W is not None:
            self.register(W)

    def register(self, W):
        [t.register(W) for t in self.triggers]

    def check(self):
        return torch.Tensor([t.check() for t in self.triggers])

    def __len__(self):
        return len(self.triggers)

    def __add__(self, other):
        return MultiTrigger(self.triggers + other.triggers)

    def __getitem__(self, index):
        return self.triggers[index]


class TriggerVector(MultiTrigger):
    def __bool__(self):
        raise RuntimeError('Boolean value of variable vector is ambiguous')


class TriggerIntersection(MultiTrigger):
    def check(self, W):
        val = super().check(W)
        return torch.all(val)


class TriggerUnion(MultiTrigger):
    def check(self, W):
        val = super().check(W)
        return torch.any(val)


class Schedule(object):
    pass


class FixedSchedule(Schedule):
    """
    A schedule that is no schedule at all. Used internally for compatibility.
    """
    pass


class OnOffSchedule(Schedule):
    """
    """
    pass
