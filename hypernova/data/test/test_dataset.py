# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Data loader tests
~~~~~~~~~~~~~~~~~
Tests of the neuroimaging data loader. These additionally require a synthetic
example dataset.
"""
import pytest
import torch
import hypernova
from pkg_resources import resource_filename as pkgrf


class TestDataLoader:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.fmriprep_dir = pkgrf('hypernova', 'examples/ds-synth')
        self.models = [
            'rps',
            'wm + csf',
            'gsr',
            'acc<v=0.7, mask=WM+CSF>'
        ]
        self.tmask = 'and(uthr0.2(d1(rps)))'
        self.ds = hypernova.data.fMRIPrepDataset(
            self.fmriprep_dir, model=self.models, tmask=self.tmask)
        self.dl = hypernova.data.dataset.ReferencedDataLoader(
            self.ds, batch_size=5, shuffle=False)

    def test_depth(self):
        assert len(self.ds.data_refs) == 10
        self.ds.set_depth(1)
        assert len(self.ds.data_refs) == 80
        self.ds.set_depth(0)

    def test_dl_depth_0(self):
        for sample in self.dl:
            break
        assert sample['images'].size() == torch.Size([5, 8, 4, 4, 4, 500])
        assert sample['rps'].size() == torch.Size([5, 8, 500, 6])
        assert sample['wm + csf'].size() == torch.Size([5, 8, 500, 2])
        assert sample['gsr'].size() == torch.Size([5, 8, 500, 1])
        assert sample['acc<v=0.7, mask=WM+CSF>'].size() == torch.Size(
            [5, 8, 500, 4])
        assert sample['t_r'].size() == torch.Size([5, 8, 1])
        assert sample['tmask'].size() == torch.Size([5, 8, 500, 1])

    def test_dl_depth_1(self):
        self.ds.set_depth(1)
        for sample in self.dl:
            break
        assert sample['images'].size() == torch.Size([5, 4, 4, 4, 500])
        assert sample['rps'].size() == torch.Size([5, 500, 6])
        assert sample['wm + csf'].size() == torch.Size([5, 500, 2])
        assert sample['gsr'].size() == torch.Size([5, 500, 1])
        assert sample['acc<v=0.7, mask=WM+CSF>'].size() == torch.Size(
            [5, 500, 3])
        assert sample['t_r'].size() == torch.Size([5, 1])
        assert sample['tmask'].size() == torch.Size([5, 500, 1])
        assert sample['tmask'][0].sum() == 333
        self.ds.set_depth(0)
