# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unit tests for toy datasets
"""
import pytest

from hypercoil.neuro.datasets import MSCMinimal
from hypercoil.functional import corr


class TestMinimalDatasets:

    def test_msc_dataset(self):
        ds = MSCMinimal(
            rms_thresh=0.2,
            shuffle=True,
            delete_if_exists=True,
            batch_size=20
        )
        batch = ds.__next__()
        out = corr(batch['bold'], weight=batch['tmask'])
        assert out.shape == (20, 400, 400)

        ds = MSCMinimal(
            rms_thresh=0.2,
            shuffle=False,
            delete_if_exists=False,
            batch_size=11
        )
        assert ds.idxmap[0] == 0
        assert ds.idxmap[-1] == 781
        batch = ds.__next__()
        out = corr(batch['bold'], weight=batch['tmask'])
        assert out.shape == (11, 400, 400)
