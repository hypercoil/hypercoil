# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for action triggers
"""
import pytest
import numpy as np
import torch
from hypercoil.engine.schedule import (
    EpochTrigger,
    LossTrigger,
    ConvergenceTrigger,
    TriggerVector
)
from hypercoil.engine.state import (
    StateVariable,
    StateIterable
)


class TestTriggers:

    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.max_epoch = 100
        self.target = torch.tensor(0.)

    def reset_state(self):
        #TODO
        # Using a dictionary here since the working state object W has not yet
        # been implemented. Need to change this when implemented.
        self.epoch = StateIterable(
            name='epoch',
            init=0,
            max_iter=self.max_epoch
        )
        self.loss = StateVariable(
            name='loss',
            track_history=True,
            track_deltas=True
        )
        self.vars = [self.epoch, self.loss]
        self.W = {v.name: v for v in self.vars}
        self.X = torch.tensor(10.0)
        self.X.requires_grad = True

    def test_epoch_trigger(self):
        self.reset_state()
        self.trigger = TriggerVector([
            EpochTrigger(10),
            EpochTrigger(0.3)
        ])
        self.trigger.register(self.W)
        for _ in self.epoch:
            if self.epoch < 10:
                assert not self.trigger[0]
                assert not self.trigger[1]
            elif self.epoch < 30:
                assert self.trigger[0]
                assert not self.trigger[1]
            else:
                assert self.trigger[0]
                assert self.trigger[1]

    def test_loss_trigger(self):
        self.reset_state()
        self.trigger = TriggerVector([
            LossTrigger(25),
            ConvergenceTrigger(1e-4)
        ])
        self.trigger.register(self.W)
        loss = torch.nn.MSELoss()
        opt = torch.optim.SGD([self.X], lr=0.1)
        for _ in self.epoch:
            self.loss.assign(loss(self.X, self.target))
            self.loss.backward()
            if self.epoch < 5:
                assert not self.trigger[0]
                assert not self.trigger[1]
            elif self.epoch < 30:
                assert self.trigger[0]
                assert not self.trigger[1]
            else:
                assert self.trigger[0]
                assert self.trigger[1]
            opt.step()
            opt.zero_grad()
