# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Conveyances
~~~~~~~~~~~
Abstractions representing transport through a model.
"""
import torch
from . import (
    SentryModule,
    SentryMessage,
    ModelArgument,
    UnpackingModelArgument
)
from .action import (
    Convey,
    CountBatches,
    BatchRelease
)


def transport(*pparams, **params):
    return (*pparams, *list(params.values()))


class BaseConveyance(SentryModule):
    def __init__(self, lines=None, line_filters=None):
        super().__init__()
        if isinstance(lines, str): lines = [lines]
        if lines is None: lines = [(None, None)]
        if line_filters is None: line_filters = {}
        lines = [l if isinstance(l, tuple) else (l, l) for l in lines]
        self.lines = lines
        self.receive = set()
        self.transmit = set()
        self.transmit_map = {}
        self.data_transmission = SentryMessage()
        self.line_filters = line_filters
        for (receive, transmit) in self.lines:
            convey = Convey(
                receive_line=receive,
                transmit_line=transmit
            )
            self.register_action(convey)
            self.receive = self.receive.union({receive})
            self.transmit = self.transmit.union({transmit})
            if self.transmit_map.get(receive) is None:
                self.transmit_map[receive] = [transmit]
            else:
                self.transmit_map[receive] += [transmit]

    def connect_downstream(self, conveyance):
        self.register_sentry(conveyance)

    def connect_upstream(self, conveyance):
        conveyance.register_sentry(self)

    def _filter_line(self, data, line):
        if self.line_filters.get(line) is None:
            return data
        arg = type(data)
        mapped = {k: v for k, v in data.items()
                  if k in self.line_filters.get(line)}
        return arg(**mapped)

    def _update_transmission(self, data, line):
        self.data_transmission.update(
            (line, self._filter_line(data, line))
        )

    def _transmit(self):
        self.message.update(
            ('DATA', self.data_transmission)
        )
        for s in self.listeners:
            s._listen(self.message)
        self.data_transmission.clear()
        self.message.clear()


class Conveyance(BaseConveyance):
    def __init__(self, influx, efflux, model=None,
                 lines=None, line_filters=None):
        super().__init__(lines=lines, line_filters=line_filters)
        if model is None:
            model = transport
        self.model = model
        self.influx = influx
        self.efflux = efflux

    def forward(self, arg, line=None):
        input = self.influx(arg)
        if isinstance(input, UnpackingModelArgument):
            output = self.model(**input)
        else:
            output = self.model(input)
        output = self.efflux(output, arg)
        for transmit_line in self.transmit_map[line]:
            self._update_transmission(output, transmit_line)
        self._transmit()
        return output


class Origin(BaseConveyance):
    def __init__(self, pipeline, lines=None, line_filters=None, **params):
        if isinstance(lines, str): lines = [lines]
        if lines is None: lines = [None]
        lines = [('source', line) for line in lines]
        super().__init__(lines=lines, line_filters=line_filters)
        self.pipeline = pipeline
        self.loader = torch.utils.data.DataLoader(
            self.pipeline, **params)
        self.iter_loader = iter(self.loader)

    def forward(self, arg=None, line=None):
        data = ModelArgument(**next(self.iter_loader))
        self.message.update(
            ('BATCH', self.loader.batch_size)
        )
        for transmit_line in self.transmit_map[line]:
            self._update_transmission(data, transmit_line)
        self._transmit()
        return data


class DataPool(BaseConveyance):
    def __init__(self, release_size=1, lines=None, line_filters=None):
        super().__init__(lines=lines, line_filters=line_filters)
        self.register_action(CountBatches())
        self.register_action(BatchRelease(batch_size=release_size))
        self.reset()
        self.batched = 0

    def reset(self):
        self.pool = {line: [] for line in self.transmit}

    def release(self):
        for line, pool in self.pool.items():
            transmit = ModelArgument()
            try:
                keys = list(pool[0].keys())
                for key in keys:
                    rebatched = [a[key] for a in pool]
                    transmit[key] = torch.cat(rebatched, dim=0)
            except IndexError: # empty pool
                return
            self._update_transmission(transmit, line)
        self._transmit()
        self.reset()

    def forward(self, arg, line=None):
        self.pool[line] += [arg]
