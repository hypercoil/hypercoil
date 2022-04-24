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
    RevolveParametersSWA,
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
        self.swa_model = swa_model
        self.model = model
        self.register_action(StochasticWeightAveraging(
            swa_start=swa_start,
            swa_scheduler=swa_scheduler,
            scheduler=scheduler
        ))


class SWAPR(SWA):
    """
    Stochastic weight averaging with parameter revolution. Almost certainly a
    terrible idea. This is really designed specifically for the parcellation/
    penalised entropy setting for reasons that will be elaborated upon. But,
    again, almost certainly a terrible idea, as it likely defeats some of the
    rationale for using SWA in the first place.
    """
    def __init__(
        self,
        epochs,
        swa_start,
        swa_model,
        swa_scheduler,
        model,
        scheduler,
        revolve_epochs
    ):
        super().__init__(
            epochs=epochs,
            swa_start=swa_start,
            swa_model=swa_model,
            swa_scheduler=swa_scheduler,
            model=model,
            scheduler=scheduler
        )
        self.register_action(RevolveParametersSWA(
            revolve_epochs=revolve_epochs
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
                x_offset = torch.tensor(e - (begin_epoch + x_scale / 2) + 0.5)
                y_offset = begin_nu
                return y_scale * torch.sigmoid(x_offset).item() + y_offset
        return end_nu


class MultiplierDecaySchedule(MultiplierTransitionSchedule):
    @staticmethod
    def get_transform(e, transitions):
        BASE_SCALE = 0.63212055882
        BASE_OFFSET = 1 - BASE_SCALE
        for ((begin_epoch, end_epoch),
             (begin_nu, end_nu)) in transitions.items():
            if e < begin_epoch:
                return begin_nu
            elif e < end_epoch:
                x_scale = (end_epoch - begin_epoch)
                y_scale = (end_nu - begin_nu) / BASE_SCALE
                x_offset = torch.tensor(e - begin_epoch)
                y_offset = end_nu
                return -y_scale * (torch.exp(-x_offset / x_scale
                    ).item() - BASE_OFFSET) + y_offset
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
