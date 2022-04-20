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


def transport(*pparams, **params):
    return (*pparams, *list(params.values()))


class BaseConveyance(SentryModule):
    def connect_downstream(self, conveyance):
        self.register_sentry(conveyance)

    def connect_upstream(self, conveyance):
        conveyance.register_sentry(self)


class Conveyance(BaseConveyance):
    def __init__(self, influx, efflux, model=None):
        super().__init__()
        if model is None:
            model = transport
        self.model = model
        self.influx = influx
        self.efflux = efflux

    def forward(self, arg):
        input = self.influx(arg)
        if isinstance(input, UnpackingModelArgument):
            output = self.model(**input)
        else:
            output = self.model(input)
        return self.efflux(output)


class Origin(BaseConveyance):
    def __init__(self, pipeline, **params):
        super().__init__()
        self.pipeline = pipeline
        self.loader = torch.utils.data.DataLoader(
            self.pipeline, **params)

    def forward(self, arg=None):
        return ModelArgument(**next(iter(self.loader)))
