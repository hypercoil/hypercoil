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


class ResetMultiplier(SentryAction):
    def __init__(self):
        super().__init__(trigger=['NU_BASE'])

    def propagate(self, sentry, received):
        for s in sentry.listeners:
            message = {'NU': received['NU_BASE']}
            s._listen(message)


class RecordLoss(SentryAction):
    def __init__(self):
        super().__init__(trigger=['LOSS', 'NAME', 'NU'])

    def propagate(self, sentry, received):
        name = received['NAME']
        sentry.epoch_buffer[name] += [received['LOSS']]
        sentry.epoch_buffer[f'{name}_norm'] += (
            [received['LOSS'] / received['NU']])


class ArchiveLoss(SentryAction):
    def __init__(self):
        super().__init__(trigger=['EPOCH'])

    def propagate(self, sentry, received):
        staging = {}
        for loss, record in sentry.epoch_buffer.items():
            if len(record) == 0:
                continue
            staging[loss] = sum(record) / len(record)
            sentry.epoch_buffer[loss] = []
        if len(staging) == 0:
            return
        for loss, archive in sentry.archive.items():
            new = staging.get(loss, float('nan'))
            sentry.archive[loss] += [new]


class ModuleReport(SentryAction):
    def __init__(self, report_interval, save_root=None,
                 save_format='.png', *args, **kwargs):
        super().__init__(trigger=['EPOCH'])
        self.report_interval = report_interval
        self.save_root = save_root
        self.save_format = save_format
        self.args = args
        self.kwargs = kwargs

    def propagate(self, sentry, received):
        if received['EPOCH'] % self.report_interval == 0:
            #TODO: we might need to revisit this save scheme for compatibility
            # with multi-output reporters
            epoch = received['EPOCH']
            if self.save_root is not None:
                save = (
                    f'{self.save_root}_epoch-{epoch}{self.save_format}'
                )
            else:
                save = None
            try:
                sentry(*self.args, save=save, **self.kwargs)
            except TypeError: #save repeated or invalid as argument
                sentry(*self.args, **self.kwargs)

    def _register_trigger(self, sentry):
        epochs_check = [isinstance(i, Epochs) for i in sentry.listening]
        if not any(epochs_check):
            sentry.actions.remove(self)
            raise ValueError(
                'Cannot register reporter action to a sentry that is not '
                'listening to epochs. Register the sentry to an epochs '
                'instance first.'
            )


##TODO: change this functionality when we've made every module a sentry
class ModuleSave(SentryAction):
    def __init__(self, save_interval, module, attribute, save_root,
                 save_format='.pt', *args, **kwargs):
        super().__init__(trigger=['EPOCH'])
        self.save_interval = save_interval
        self.module = module
        self.attribute = attribute
        self.save_root = save_root
        self.save_format = save_format
        self.args = args
        self.kwargs = kwargs

    def propagate(self, sentry, received):
        if received['EPOCH'] % self.save_interval == 0:
            epoch = received['EPOCH']
            to_save = [v for k, v in self.module.named_modules()
                       if k == self.attribute]
            if not to_save:
                raise AttributeError(
                    f'Module {self.module} has no submodule {self.attribute}'
                )
            save_path = (
                f'{self.save_root}_epoch-{epoch}{self.save_format}'
            )
            torch.save(to_save, save_path)


class WriteTSV(SentryAction):
    def __init__(self, save_interval, save_root, overwrite=True):
        super().__init__(trigger=['EPOCH'])
        self.save_interval = save_interval
        self.save_root = save_root
        self.overwrite = overwrite

    def propagate(self, sentry, received):
        if received['EPOCH'] % self.save_interval == 0:
            to_save = sentry.data
            if self.overwrite:
                save_path = f'{self.save_root}.tsv'
            else:
                epoch = received['EPOCH']
                save_path = (
                    f'{self.save_root}_epoch-{epoch}.tsv'
                )
            to_save.to_csv(save_path, index=False, sep='\t')


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


class SchedulerSentry(Sentry):
    def __init__(self, epochs, base=1):
        super().__init__()
        self.base = base
        epochs.register_sentry(self)
        self.register_action(ResetMultiplier())

    def reset(self):
        for action in self.actions:
            action({'NU_BASE': self.base})


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
    def __init__(self, epochs, save_interval=None, save_root=None):
        super().__init__()
        self.archive = {}
        self.epoch_buffer = {}
        self.register_action(RecordLoss())
        self.register_action(ArchiveLoss())
        epochs.register_sentry(self)

        if save_interval is not None:
            self.register_action(WriteTSV(
                save_interval=save_interval,
                save_root=save_root,
                overwrite=True
            ))

    def _register_trigger(self, sentry):
        #TODO: explicitly check for Loss when we separate implementations
        # from base classes.
        if isinstance(sentry, SentryModule):
            self.archive[sentry.name] = []
            self.archive[f'{sentry.name}_norm'] = []
            self.epoch_buffer[sentry.name] = []
            self.epoch_buffer[f'{sentry.name}_norm'] = []

    def get(self, name, normalised=False):
        if normalised:
            return self.archive[f'{name}_norm']
        else:
            return self.archive[name]

    def to_df(self):
        return pd.DataFrame(self.archive)

    @property
    def data(self):
        return self.to_df()
