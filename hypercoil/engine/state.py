# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Learning state
~~~~~~~~~~~~~~
Environment to track values and histories related to learning.
"""
import torch


class StateVariable(object):
    def __init__(self, name='state', init=None,
                 track_history=False, track_deltas=False):
        self.name = name
        self.assignment = None
        self.track_history = track_history
        self.track_deltas = track_deltas
        if init is not None:
            self.assign(init)
        if self.track_history:
            self._history = []
        if self.track_deltas:
            self._delta = []

    def assign(self, val):
        if not isinstance(val, torch.Tensor):
            val = torch.tensor(val)
        if self.track_history:
            if self.track_deltas and len(self._history) > 0:
                self._delta += [val - self._history[-1]]
            self._history += [val.clone().detach()]
        self.assignment = val

    def backward(self):
        self.assignment.backward()

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return self.value != other

    def __ge__(self, other):
        return self.value >= other

    def __le__(self, other):
        return self.value <= other

    def __gt__(self, other):
        return self.value > other

    def __lt__(self, other):
        return self.value < other

    def __repr__(self):
        inside = ', '.join(self._inside())
        return f'{self.name}={type(self).__name__}({inside})'

    def _inside(self):
        inside = []
        if self.assignment is not None:
            inside += [f'value={self.assignment}']
        if self.track_history:
            inside += ['history=True']
        if self.track_deltas:
            inside += ['deltas=True']
        return inside

    @property
    def value(self):
        return self.assignment

    @property
    def history(self):
        return torch.Tensor(self._history)

    @property
    def delta(self):
        return torch.Tensor(self._delta)


class StateIterable(StateVariable):
    def __init__(self, max_iter, name='state', init=0,
                 track_history=False, track_deltas=False):
        super().__init__(name=name, init=init,
                         track_history=track_history,
                         track_deltas=track_deltas)
        self.max_iter = max_iter

    def _inside(self):
        inside = super()._inside()
        inside += [f'max={self.max_iter}']
        return inside

    def __iter__(self):
        return self

    def __next__(self):
        if self.assignment < self.max_iter:
            self.assign(self.assignment + 1)
        else:
            raise StopIteration
