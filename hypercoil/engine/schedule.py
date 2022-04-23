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
from collections.abc import Iterable
from .sentry import Sentry
from .action import (
    StepScheduler,
    LossStepScheduler,
    StochasticWeightAveraging,
    RecordLoss,
    WeightDecayStep,
    ResetMultiplier,
    PropagateMultiplierFromEpochTransform,
    PropagateMultiplierFromRecursiveTransform
)


class _Schedule(Sentry):
    def __init__(self, epochs):
        super().__init__()
        epochs.register_sentry(self)


##TODO: track the last seen epoch, and if the next one is unexpected,
# fast-forward steps to the correct one. This probably belongs in the action
# rather than here.
class LRSchedule(_Schedule):
    """
    .. warning::
        Do not assign the same torch ``Scheduler`` to more than one schedule.
        This will result in multiple learning rate steps per epoch. Be advised
        that ``SWA`` is also a schedule.
    """
    def __init__(self, epochs, schedulers):
        super().__init__(epochs)
        if not isinstance(schedulers, Iterable):
            schedulers = [schedulers]
        self.schedulers = schedulers
        self.register_action(StepScheduler())


class LRLossSchedule(_Schedule):
    def __init__(self, epochs, loss, schedulers):
        super().__init__(epochs)
        if not isinstance(schedulers, Iterable):
            schedulers = [schedulers]
        self.schedulers = schedulers
        self.epoch_buffer = {}
        self.register_action(RecordLoss())
        self.register_action(LossStepScheduler())

    def _register_trigger(self, sentry):
        #TODO: explicitly check for Loss when we configure all loss variants
        # to subclass something.
        if isinstance(sentry, SentryModule):
            self.epoch_buffer[sentry.name] = []
            self.epoch_buffer[f'{sentry.name}_norm'] = []


class SWA(_Schedule):
    def __init__(
        self,
        epochs,
        swa_start,
        swa_model,
        swa_scheduler,
        model,
        scheduler
    ):
        super().__init__(epochs)
        self.register_action(StochasticWeightAveraging(
            swa_start=swa_start,
            swa_model=swa_model,
            swa_scheduler=swa_scheduler,
            model=model,
            scheduler=scheduler
        ))


class WeightDecayMultiStepSchedule(_Schedule):
    def __init__(self, epochs, steps, param_groups):
        super().__init__(epochs)
        if not isinstance(param_groups, Iterable):
            param_groups = [param_groups]
        self.param_groups = param_groups
        self.register_action(WeightDecayStep(steps=steps))


class _MultiplierSchedule(_Schedule):
    def __init__(self, epochs, base=1):
        super().__init__(epochs)
        self.base = base
        self.register_action(ResetMultiplier())

    def reset(self):
        self.message.clear()
        self.message.update(('NU_BASE', self.base))
        for action in self.actions:
            action(self.message)
        self.message.clear()


class MultiplierSchedule(_MultiplierSchedule):
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


class MultiplierCascadeSchedule(MultiplierTransitionSchedule):
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


class MultiplierRampSchedule(MultiplierTransitionSchedule):
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
