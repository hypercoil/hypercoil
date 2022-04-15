# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Diff terminals
~~~~~~~~~~~~~~
Differentiable program terminals. Currently minimal functionality.
"""
from .sentry import SentryModule


class Terminal(SentryModule):
    """
    Right now, this does nothing whatsoever.
    """
    def __init__(self, loss, arg_factory=None):
        super().__init__()
        self.loss = loss
        self.name = self.loss.name

    def _transmit(self, loss_value):
        self.message.update(
            ('NAME', self.loss.name),
            ('LOSS', loss_value.detach().item()),
            ('NU', self.loss.nu)
        )
        for s in self.listeners:
            s._listen(self.message)
        self.message.clear()

    def forward(self, arg):
        loss = self.loss(**arg)
        self._transmit(loss)
        return loss


class ReactiveTerminal(Terminal):
    """
    Right now, this is just an abstraction to handle salami slicing when the
    loss operation over the entire input tensor is too large to fit into
    memory.

    Note that the forward call immediately triggers a series of backward
    calls in order to reduce the memory footprint.
    """
    def __init__(self, loss, slice_target, slice_axis, max_slice,
                 normalise_by_len=True, pretransforms=None):
        super().__init__(loss=loss)
        self.slice_target = slice_target
        self.slice_axis = slice_axis
        self.max_slice = max_slice
        self.normalise_by_len = normalise_by_len
        self.pretransforms = pretransforms or {}

    def forward(self, arg):
        pretransform = {}
        for k in self.pretransforms.keys():
            if k != self.slice_target:
                pretransform[k] = arg.__getitem__(k)
        slice_target = arg.__getitem__(self.slice_target)
        slc = [slice(None) for _ in range(slice_target.dim())]
        total = slice_target.shape[self.slice_axis]
        begin = 0
        loss = 0
        while begin < total:
            end = begin + self.max_slice
            slc[self.slice_axis] = slice(begin, end)
            sliced = slice_target[slc]
            arg.__setitem__(self.slice_target, sliced)
            for k, v in self.pretransforms.items():
                if k != self.slice_target:
                    arg.__setitem__(k, v(pretransform[k]))
                else:
                    arg.__setitem__(k, v(sliced))
            Y = self.loss(**arg)
            if self.normalise_by_len:
                l = sliced.shape[self.slice_axis]
                Y = Y * l / total
            loss += Y
            Y.backward()
            begin += self.max_slice
        self._transmit(loss)
        return loss


class ReactiveMultiTerminal(Terminal):
    """
    Generalised `ReactiveTerminal`.

    `slice_instructions` is a dict whose keys correspond to `slice_target`
    in `ReactiveTerminal` and whose values correspond to `slice_axis`.
    """
    def __init__(self, loss, slice_instructions, max_slice,
                 normalise_by_len=True, pretransforms=None):
        super().__init__(loss=loss)
        self.slice_instructions = slice_instructions
        self.max_slice = max_slice
        self.normalise_by_len = normalise_by_len
        self.pretransforms = pretransforms or {}

    def forward(self, arg):
        instructions = {}
        targets ={}
        totals = []
        pretransform = {}
        for k in self.pretransforms.keys():
            if k not in self.slice_instructions.keys():
                pretransform[k] = arg.__getitem__(k)
        for k, v in self.slice_instructions.items():
            targets[k] = arg.__getitem__(k)
            slc = [slice(None) for _ in range(targets[k].dim())]
            totals += [targets[k].shape[v]]
            instructions[k] = (slc, v)
        total = totals[0]
        assert all([t == total for t in totals])
        begin = 0
        loss = 0
        while begin < total:
            end = begin + self.max_slice
            for k, (slc, v) in instructions.items():
                slice_target = targets[k]
                slc[v] = slice(begin, end)
                sliced = slice_target[slc]
                arg.__setitem__(k, sliced)
                l = sliced.shape[v] # wasteful but cleaner this way
            for k, v in self.pretransforms.items():
                if k not in self.slice_instructions.keys():
                    arg.__setitem__(k, v(pretransform[k]))
                else:
                    arg.__setitem__(k, v(arg[k]))
            Y = self.loss(**arg)
            if self.normalise_by_len:
                Y = Y * l / total
            loss += Y
            Y.backward()
            begin += self.max_slice
        self._transmit(loss)
        return loss
