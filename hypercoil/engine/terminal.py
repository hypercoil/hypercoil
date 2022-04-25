# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Diff terminals
~~~~~~~~~~~~~~
Differentiable program terminals. Currently minimal functionality.
"""
from ..functional import conform_mask
from .conveyance import Conveyance
from .argument import UnpackingModelArgument


class Terminal(Conveyance):
    def __init__(
        self,
        loss,
        args=None,
        lines=None,
        influx=None,
        argbase=None,
        retain_graph=False,
        receive_filters=None
    ):
        super().__init__(
            lines=lines,
            transmit_filters=None,
            receive_filters=receive_filters
        )
        self.loss = loss
        self.args = args
        self.name = self.loss.name
        self.influx = influx or (lambda x: x)
        self.argbase = argbase
        self.retain_graph = True
        self.reset()

    def _transmit(self, loss_value):
        from ..loss import LossScheme
        # This is a terminal. It conveys data no further.
        if not isinstance(self.loss, LossScheme):
            self.message.update(
                ('NAME', self.loss.name),
                ('LOSS', loss_value.detach().item()),
                ('NU', self.loss.nu)
            )
            for s in self.listeners:
                s._listen(self.message)
        self.reset()

    def reset(self):
        self.message.clear()
        argtype = type(self.argbase)
        self.arg = argtype(**self.argbase)

    def release(self):
        if isinstance(self.arg, UnpackingModelArgument):
            loss = self.loss(**self.arg)
        else:
            loss = self.loss(self.arg)
        self._transmit(loss)
        loss.backward(retain_graph=self.retain_graph)
        return loss

    def forward(self, arg, line=None):
        input = self.influx(self._filter_received(arg, line))
        self.arg.update(**input)
        for arg in self.args:
            if arg not in self.arg:
                return
        loss = self.release()
        return loss


##TODO: when this extreme development period is over, we need to refactor this
# and split terminal-dependent functionality from non-dependent.
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
        super().__init__(loss=loss, retain_graph=False)
        self.slice_target = slice_target
        self.slice_axis = slice_axis
        self.max_slice = max_slice
        self.normalise_by_len = normalise_by_len
        self.pretransforms = pretransforms or {}

    def forward(self, arg, axis_mask=None):
        pretransform = {}
        for k in self.pretransforms.keys():
            if k != self.slice_target:
                pretransform[k] = arg.__getitem__(k)
        slice_target = arg.__getitem__(self.slice_target)
        slc = [slice(None) for _ in range(slice_target.dim())]
        total = slice_target.shape[self.slice_axis]
        if axis_mask is not None:
            axis_mask = conform_mask(
                slice_target,
                axis_mask,
                self.slice_axis,
                batch=True
            )
            slice_target = slice_target * axis_mask
            norm_fac = axis_mask.sum()
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
            if self.normalise_by_len and axis_mask is not None:
                mask_sliced = axis_mask[slc]
                l = mask_sliced.sum()
                Y = Y * l / norm_fac
            elif self.normalise_by_len:
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
