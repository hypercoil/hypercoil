# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Diff terminals
~~~~~~~~~~~~~~
Differentiable program terminals. Currently minimal functionality.
"""
from torch.nn import Module


class Terminal(Module):
    """
    Right now, this does nothing whatsoever.
    """
    def __init__(self, loss, arg_factory=None):
        super().__init__()
        self.loss = loss

    def forward(self, arg):
        return self.loss(arg)


class ReactiveTerminal(Terminal):
    """
    Right now, this is just an abstraction to handle salami slicing when the
    loss operation over the entire input tensor is too large to fit into
    memory.

    Note that the forward call immediately triggers a series of backward
    calls in order to reduce the memory footprint.
    """
    def __init__(self, loss, slice_target, slice_axis, max_slice,
                 normalise_by_len=True):
        super().__init__(loss=loss)
        self.slice_target = slice_target
        self.slice_axis = slice_axis
        self.max_slice = max_slice
        self.normalise_by_len = normalise_by_len

    def forward(self, arg):
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
            Y = self.loss(**arg)
            if self.normalise_by_len:
                l = sliced.shape[self.slice_axis]
                Y = Y * l / total
            loss += Y
            Y.backward()
            begin += self.max_slice
        return loss
