# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for sentry objects.
"""
import pytest
import torch
import numpy as np
from hypercoil.engine import (
    Epochs,
    MultiplierTransformSchedule,
    MultiplierRecursiveSchedule,
    MultiplierLinearSchedule,
    MultiplierSigmoidSchedule,
    LossArchive
)
from hypercoil.engine.sentry import (
    SentryAction
)
from hypercoil.loss import (
    SoftmaxEntropy
)


class TestSentry:

    def test_multiplier_schedule(self):

        class IncompleteTransmit(SentryAction):
            def __init__(self):
                super().__init__(trigger=['EPOCH'])

            def propagate(self, sentry, received):
                message = {'LOSS': -1000, 'NAME': 'BlubbaTheWhale'}
                for s in sentry.listeners:
                    s._listen(message)

        max_epoch = 100
        epochs = Epochs(max_epoch)
        schedule0 = MultiplierTransformSchedule(
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
        schedule3 = MultiplierLinearSchedule(
            epochs=epochs,
            transitions={(10, 30): 5,
                         (40, 50): 3,
                         (55, 60): -1,
                         (75, 90): 11},
            base=1
        )
        bad_transmitter = IncompleteTransmit()
        schedule0.register_action(bad_transmitter)
        archive = LossArchive(epochs)
        loss0 = SoftmaxEntropy(nu=schedule0, name='loss0')
        loss1 = SoftmaxEntropy(nu=schedule1, name='loss1')
        loss2 = SoftmaxEntropy(nu=schedule2, name='loss2')
        loss3 = SoftmaxEntropy(nu=schedule3, name='loss3')
        loss0.register_sentry(archive)
        loss1.register_sentry(archive)
        loss2.register_sentry(archive)
        loss3.register_sentry(archive)
        schedule0.register_sentry(archive) # test ignoring irrelevant messages
        Z = torch.rand(10)
        reset = False

        for e in epochs:
            loss0(Z)
            loss1(Z)
            loss2(Z)
            loss3(Z)
            if e == 95 and not reset:
                epochs.reset()
                schedule1.reset()
                reset = True

        begin = archive.get('loss0')[0]
        end = archive.get('loss0')[-1]

        assert np.isclose((begin * 1.01 ** (max_epoch - 1)), loss0(Z))
        assert end == archive.get('loss1')[-1]
        assert np.isclose(loss0(Z) / 1.01, end)

        loss0_tape = archive.get('loss0', normalised=True)
        assert np.allclose(loss0_tape[0], loss0_tape)

        orig = archive.get('loss2')[0]
        assert np.isclose(archive.archive['loss2'][20], 1 * orig)
        assert np.isclose(archive.archive['loss2'][30], 3 * orig)
        assert np.isclose(archive.archive['loss2'][40], 5 * orig)
        assert np.isclose(archive.archive['loss2'][60], 5 * orig)
        assert np.isclose(archive.archive['loss2'][70], 3.5 * orig)
        assert np.isclose(archive.archive['loss2'][80], 2 * orig)

        orig = archive.get('loss3')[0]
        assert np.isclose(archive.archive['loss3'][10], 1 * orig)
        assert np.isclose(archive.archive['loss3'][20], 3 * orig)
        assert np.isclose(archive.archive['loss3'][40], 5 * orig)
        assert np.isclose(archive.archive['loss3'][50], 3 * orig)
        assert np.isclose(archive.archive['loss3'][60], -1 * orig)
        assert np.isclose(archive.archive['loss3'][70], -1 * orig)
        assert np.isclose(archive.archive['loss3'][80], 3 * orig)
        assert np.isclose(archive.archive['loss3'][90], 11 * orig)
