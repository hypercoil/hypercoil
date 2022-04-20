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
    ModelArgument,
    UnpackingModelArgument
)
from .actions import Convey


def transport(*pparams, **params):
    return (*pparams, *list(params.values()))


class BaseConveyance(SentryModule):
    def __init__(self, lines=None):
        super().__init__()
        if lines is None: lines = [(None, None)]
        lines = [l if isinstance(l, tuple) else (l, l) for l in lines]
        self.lines = lines
        for (receive, transmit) in self.lines:
            convey = Convey(
                receive_line=receive,
                transmit_line=transmit
            )
            self.register_action(convey)

    def connect_downstream(self, conveyance):
        self.register_sentry(conveyance)

    def connect_upstream(self, conveyance):
        conveyance.register_sentry(self)

    def _transmit(self, data, line):
        self.message.update(
            ('DATA', {line : data})
        )
        for s in self.listeners:
            s._listen(self.message)
        self.message.clear()


class Conveyance(BaseConveyance):
    def __init__(self, influx, efflux, model=None, lines=None):
        super().__init__(lines)
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
        output = self.efflux(output)
        self._transmit(output, line)
        return output


class Origin(BaseConveyance):
    def __init__(self, pipeline, lines=None, **params):
        super().__init__(lines)
        self.pipeline = pipeline
        self.loader = torch.utils.data.DataLoader(
            self.pipeline, **params)

    def forward(self, arg=None, line=None):
        data = ModelArgument(**next(iter(self.loader)))
        self._transmit(data, line)
        return data
