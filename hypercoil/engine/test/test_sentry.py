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
    MultiplierSigmoidSchedule,
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
        schedule2 = MultiplierSigmoidSchedule(
            epochs=epochs,
            transitions={(21, 40): 5, (61, 80): 2},
            base=1
        )
        archive = LossArchive()
        loss0 = SoftmaxEntropy(nu=schedule0, name='loss0')
        loss1 = SoftmaxEntropy(nu=schedule1, name='loss1')
        loss2 = SoftmaxEntropy(nu=schedule2, name='loss2')
        loss0.register_sentry(archive)
        loss1.register_sentry(archive)
        loss2.register_sentry(archive)
        Z = torch.rand(10)

        begin = loss0(Z)

        for e in epochs:
            loss0(Z)
            loss1(Z)
            loss2(Z)

        end = loss0(Z)

        assert (begin * 1.01 ** (max_epoch - 1)) == loss0(Z)
        assert loss0(Z) == loss1(Z)
        assert loss0(Z) == archive.get('loss0')[-1]

        loss0_tape = archive.get('loss0', normalised=True)
        assert np.allclose(loss0_tape[0], loss0_tape)

        assert archive.archive['loss2_nu'][20] == 1
        assert archive.archive['loss2_nu'][30] == 3
        assert archive.archive['loss2_nu'][40] == 5
        assert archive.archive['loss2_nu'][60] == 5
        assert archive.archive['loss2_nu'][70] == 3.5
        assert archive.archive['loss2_nu'][80] == 2
