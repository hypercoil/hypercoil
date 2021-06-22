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
