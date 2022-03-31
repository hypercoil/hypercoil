# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for sentry objects.
"""
import pytest
import torch
from hypercoil.engine.sentry import (
    Epochs,
    MultiplierSchedule,
    LossArchive
)
from hypercoil.loss import (
    SoftmaxEntropy
)


class TestSentry:

    def test_multiplier_schedule(self):

        max_epoch = 100
        epochs = Epochs(max_epoch)
        schedule = MultiplierSchedule(
            epochs=epochs,
            transform=lambda e: 1.01 ** e
        )
        archive = LossArchive()
        loss = SoftmaxEntropy(nu=schedule)
        loss.register_sentry(archive)
        Z = torch.rand(10)

        begin = loss(Z)

        for e in epochs:
            loss(Z)

        end = loss(Z)

        assert (begin * 1.01 ** (max_epoch - 1)) == loss(Z)
