# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Minimal MSC Dataset
~~~~~~~~~~~~~~~~~~~
Test the minimal MSC dataset.
"""
import jax
from hypercoil.functional import corr
from hypercoil.neuro.data_msc import get_msc_dir, MSCMinimal


class TestMinimalMSC:

    def test_msc_dataset(self):
        key = jax.random.PRNGKey(0)
        ds = MSCMinimal(
            rms_thresh=0.2,
            shuffle=True,
            delete_if_exists=True,
            batch_size=20,
            key=key,
        )
        batch = ds.__next__()
        out = corr(batch['bold'], weight=batch['tmask'].astype('float32'))
        assert out.shape == (20, 400, 400)

        ds_dir = get_msc_dir()
        assert ds_dir == ds.srcdir

        ds = MSCMinimal(
            rms_thresh=0.2,
            shuffle=False,
            delete_if_exists=False,
            batch_size=11,
        )
        assert ds.idxmap[0] == 0
        assert ds.idxmap[-1] == 781
        batch = ds.__next__()
        out = corr(batch['bold'], weight=batch['tmask'].astype('float32'))
        assert out.shape == (11, 400, 400)

        ds = MSCMinimal(
            rms_thresh=0.2,
            shuffle=True,
            delete_if_exists=False,
            batch_size=3,
            sub='01',
            ses='01',
            task='rest',
            key=key,
        )
        for epoch in range(5):
            ds.cfg_iter(shuffle=True, key=jax.random.fold_in(key, epoch))
            for batch in iter(ds):
                assert batch['bold'].shape == (3, 400, 814)
