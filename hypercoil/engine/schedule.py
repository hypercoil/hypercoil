# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Learning schedules
~~~~~~~~~~~~~~~~~~
Hyperparameter schedule sentries for toggling and changing hyperparameters
as the program learns.
"""
import torch
from functools import partial
from .sentry import Sentry
from .action import (
    ResetMultiplier,
    PropagateMultiplierFromEpochTransform,
    PropagateMultiplierFromRecursiveTransform
)


class SchedulerSentry(Sentry):
    def __init__(self, epochs, base=1):
        super().__init__()
        self.base = base
        epochs.register_sentry(self)
        self.register_action(ResetMultiplier())

    def reset(self):
        self.message.clear()
        self.message.update(('NU_BASE', self.base))
        for action in self.actions:
            action(self.message)
        self.message.clear()


class MultiplierSchedule(SchedulerSentry):
    def __init__(self, epochs, transform, base=1):
        super().__init__(epochs=epochs, base=base)
        self.transform=transform


class MultiplierTransformSchedule(MultiplierSchedule):
    def __init__(self, epochs, transform, base=1):
        super().__init__(epochs=epochs, transform=transform, base=base)
        self.register_action(PropagateMultiplierFromEpochTransform(
            transform=self.transform
        ))


class MultiplierRecursiveSchedule(MultiplierSchedule):
    def __init__(self, epochs, transform, base=1):
        super().__init__(epochs=epochs, transform=transform, base=base)
        self.register_action(PropagateMultiplierFromRecursiveTransform(
            transform=self.transform
        ))


class MultiplierTransitionSchedule(MultiplierTransformSchedule):
    def __init__(self, epochs, transitions, base=1):
        cur = base
        for k, v in transitions.items():
            transitions[k] = (cur, v)
            cur = v
        transform = partial(type(self).get_transform,
                            transitions=transitions)
        super().__init__(
            epochs=epochs,
            transform=transform,
            base=base
        )


class MultiplierSigmoidSchedule(MultiplierTransitionSchedule):
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


class MultiplierLinearSchedule(MultiplierTransitionSchedule):
    @staticmethod
    def get_transform(e, transitions):
        for ((begin_epoch, end_epoch),
             (begin_nu, end_nu)) in transitions.items():
            if e < begin_epoch:
                return begin_nu
            elif e < end_epoch:
                slope = (end_nu - begin_nu) / (end_epoch - begin_epoch)
                start = begin_nu
                offset = (e - begin_epoch)
                return start + slope * offset
        return end_nu
