# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Reporters
~~~~~~~~~
Reporter sentries for relaying and communicating results and parameter changes
over the course of learning.
"""
from .sentry import Sentry, SentryModule
from .action import (
    RecordLoss,
    ArchiveLoss,
    WriteTSV
)


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
