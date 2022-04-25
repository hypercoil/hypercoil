# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Conveyances
~~~~~~~~~~~
Abstractions representing transport through a model.
"""
import torch
import warnings
from . import (
    SentryModule,
    SentryMessage,
    ModelArgument,
    UnpackingModelArgument
)
from .action import (
    Convey,
    CountBatches,
    BatchRelease,
    ConfluxRelease
)


def transport(*pparams, **params):
    return (*pparams, *list(params.values()))


class BaseConveyance(SentryModule):
    def __init__(
        self,
        lines=None,
        transmit_filters=None,
        receive_filters=None
    ):
        super().__init__()
        if transmit_filters is None: transmit_filters = {}
        if receive_filters is None: receive_filters = {}
        lines = self._fmt_lines(lines)
        self.lines = set()
        self.receive = set()
        self.transmit = set()
        self.transmit_map = {}
        self.data_transmission = SentryMessage()
        self.transmit_filters = transmit_filters
        self.receive_filters = receive_filters
        for line in lines:
            self.add_line(line)

    def _fmt_lines(self, lines):
        if isinstance(lines, str): lines = [lines]
        if lines is None: lines = [None]
        lines = [l if isinstance(l, tuple) else (l, l) for l in lines]
        return lines

    def _add_line_extra(self, line):
        pass

    def add_line(self, line):
        if not isinstance(line, tuple):
            line = (line, line)
        if line in self.lines:
            return
        receive, transmit = line
        convey = Convey(
            line=receive
        )
        self.lines = self.lines.union({(receive, transmit)})
        self.register_action(convey)
        self.receive = self.receive.union({receive})
        self.transmit = self.transmit.union({transmit})
        if self.transmit_map.get(receive) is None:
            self.transmit_map[receive] = [transmit]
        else:
            self.transmit_map[receive] += [transmit]
        self._add_line_extra(line)

    def transmit_filter_cfg(self, line, filter):
        self.transmit_filters[line] = filter

    def receive_filter_cfg(self, line, filter):
        self.receive_filters[line] = filter

    def connect_downstream(self, conveyance):
        self.register_sentry(conveyance)

    def connect_upstream(self, conveyance):
        conveyance.register_sentry(self)

    def _filter_line(self, data, line, filters):
        if filters.get(line) is None:
            return data
        arg = type(data)
        mapped = {k: v for k, v in data.items()
                  if k in filters.get(line)}
        return arg(**mapped)

    def _filter_received(self, data, line):
        return self._filter_line(data, line, self.receive_filters)

    def _filter_transmission(self, data, line):
        return self._filter_line(data, line, self.transmit_filters)

    def _update_transmission(self, data, line):
        self.data_transmission.update(
            (line, self._filter_transmission(data, line))
        )

    def clear_transmission(self):
        self.data_transmission.clear()

    def _update_all_transmissions(self, data, line, transmit_lines):
        for transmit_line in self.transmit_map[line]:
            if transmit_lines is None or transmit_line in transmit_lines:
                self._update_transmission(data, transmit_line)

    def _transmit(self):
        self.message.update(
            ('DATA', self.data_transmission)
        )
        for s in self.listeners:
            #print(self, s, self.message)
            s._listen(self.message)
        self.data_transmission.clear()
        self.message.clear()


class Conveyance(BaseConveyance):
    def __init__(self, influx=None, efflux=None, model=None,
                 lines=None, transmit_filters=None,
                 receive_filters=None):
        super().__init__(
            lines=lines,
            transmit_filters=transmit_filters,
            receive_filters=receive_filters
        )
        if model is None:
            model = transport
        self.model = model
        self.influx = influx or (lambda x: x)
        self.efflux = efflux or (
            lambda output, arg: ModelArgument(output=output))

    def forward(self, arg, line=None, transmit_lines=None):
        input = self.influx(self._filter_received(arg, line))
        if isinstance(input, UnpackingModelArgument):
            output = self.model(**input)
        else:
            output = self.model(input)
        output = self.efflux(output=output, arg=arg)
        self._update_all_transmissions(output, line, transmit_lines)
        self._transmit()
        return output


class Origin(BaseConveyance):
    def __init__(
        self,
        pipeline,
        lines=None,
        efflux=None,
        transmit_filters=None,
        receive_filters=None,
        **params
    ):
        if isinstance(lines, str): lines = [lines]
        if lines is None: lines = [None]
        lines = [('source', line) for line in lines]
        super().__init__(
            lines=lines, transmit_filters=transmit_filters)
        self.pipeline = pipeline
        self.efflux = efflux or (lambda arg: arg)
        self.loader = torch.utils.data.DataLoader(
            self.pipeline, **params)
        self.iter_loader = iter(self.loader)

    def forward(self, arg=None, line=None, transmit_lines=None):
        try:
            data = ModelArgument(**next(self.iter_loader))
        except StopIteration:
            self.iter_loader = iter(self.loader)
            data = ModelArgument(**next(self.iter_loader))
        data = self.efflux(self._filter_received(data, line))
        self.message.update(
            ('BATCH', self.loader.batch_size)
        )
        self._update_all_transmissions(data, line, transmit_lines)
        self._transmit()
        return data


class Hollow(BaseConveyance):
    def __init__(
        self,
        fields,
        lines=None,
        transmit_filters=None,
        receive_filters=None
    ):
        super().__init__(
            lines=lines,
            transmit_filters=transmit_filters,
            receive_filters=receive_filters
        )
        self.fields = fields

    def hollow_fields(self, arg):
        for field in self.fields:
            if arg.get(field) is None:
                new = {field : None}
                arg = ModelArgument(**arg, **new)
        return arg

    def forward(self, arg=None, line=None, transmit_lines=None):
        if arg is not None:
            data = ModelArgument(**self._filter_received(arg, line))
        else:
            data = ModelArgument()
        data = self.hollow_fields(data)
        self._update_all_transmissions(data, line, transmit_lines)
        self._transmit()


class Conflux(BaseConveyance):
    def __init__(
        self,
        fields,
        lines=None,
        transmit_filters=None,
        receive_filters=None
    ):
        super().__init__(
            lines=lines,
            transmit_filters=transmit_filters,
            receive_filters=receive_filters
        )
        self.register_action(ConfluxRelease(requirements=fields))
        self.reset()

    def reset(self):
        self.staged = ModelArgument()

    def compile(self):
        transmit = ModelArgument(**self.staged)
        for transmit_line in self.transmit:
            self._update_transmission(transmit, transmit_line)

    def release(self):
        self.compile()
        self.reset()
        self._transmit()

    def forward(self, arg, line=None):
        self.staged.update(**self._filter_received(arg, line))


class DataPool(BaseConveyance):
    def __init__(
        self,
        release_size=float('inf'),
        lines=None,
        transmit_filters=None,
        receive_filters=None
    ):
        super().__init__(
            lines=lines,
            transmit_filters=transmit_filters,
            receive_filters=receive_filters
        )
        self.batched = 0
        self.register_action(CountBatches())
        self.register_action(BatchRelease(batch_size=release_size))
        self.reset()
        if len(self.receive) > 1:
            warnings.warn(
                'DataPool receiving over multiple lines: '
                'Loss of pool size synchrony is possible. '
                'It is recommended that you create a separate '
                'DataPool for each receiving line.'
            )

    def reset(self):
        self.pool = {line: [] for line in self.transmit}

    def compile(self, inplace=False):
        for line, pool in self.pool.items():
            transmit = ModelArgument()
            try:
                keys = list(pool[0].keys())
                for key in keys:
                    rebatched = [a[key] for a in pool]
                    if isinstance(rebatched[0], torch.Tensor):
                        rebatched = torch.cat(rebatched, dim=0)
                    transmit[key] = rebatched
            except IndexError: # empty pool
                continue
            if inplace:
                self.pool[line] = [transmit]
            else:
                self._update_transmission(transmit, line)

    def sample(
        self,
        sample_size=1,
        destroy=True,
        destroy_unused=False,
        batch_key=None,
        line=None
    ):
        self.compile(inplace=True)
        ret = ModelArgument()
        repool = ModelArgument()
        sample, = self.pool[line]
        keys = list(sample.keys())
        batch_key = batch_key or keys[0]
        sampled_size = sample[batch_key].shape[0]
        require_repool = False
        restore_size = 0
        if sampled_size < sample_size:
            raise ValueError(
                f'Insufficient pooled data for a '
                f'sample of size {sample_size}')
        for k, v in sample.items():
            if sampled_size > sample_size and not destroy_unused:
                restore = v[sample_size:]
                restore_size = restore.shape[0]
                if destroy:
                    require_repool = True
                    repool[k] = restore
            ret[k] = v[:sample_size]
        if destroy:
            if require_repool:
                self.pool[line] = [repool]
            else:
                self.pool[line] = []
            self.batched = restore_size
        return ret

    def release(self):
        self.compile()
        self.reset()
        self._transmit()

    def forward(self, arg, line=None):
        self.pool[line] += [self._filter_received(arg, line)]
