# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for sentry objects.
"""
import pytest
import torch
import numpy as np
from hypercoil.engine.sentry import (
    Epochs,
    MultiplierSchedule,
    MultiplierRecursiveSchedule,
    LossArchive
)
from hypercoil.loss import (
    SoftmaxEntropy
)


class TestSentry:

    def test_multiplier_schedule(self):

        max_epoch = 100
        epochs = Epochs(max_epoch)
        schedule0 = MultiplierSchedule(
            epochs=epochs,
            transform=lambda e: 1.01 ** e
        )
        schedule1 = MultiplierRecursiveSchedule(
            epochs=epochs,
            transform=lambda nu: 1.01 * nu,
            base=(1 / 1.01)
        )
        archive = LossArchive()
        loss0 = SoftmaxEntropy(nu=schedule0, name='loss0')
        loss1 = SoftmaxEntropy(nu=schedule1, name='loss1')
        loss0.register_sentry(archive)
        loss1.register_sentry(archive)
        Z = torch.rand(10)

        begin = loss0(Z)

        for e in epochs:
            loss0(Z)
            loss1(Z)

        end = loss0(Z)

        assert (begin * 1.01 ** (max_epoch - 1)) == loss0(Z)
        assert loss0(Z) == loss1(Z)
        assert loss0(Z) == archive.get('loss0')[-1]

        loss0_tape = archive.get('loss0', normalised=True)
        assert np.allclose(loss0_tape[0], loss0_tape)
