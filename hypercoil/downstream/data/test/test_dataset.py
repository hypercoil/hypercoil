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
import hypercoil
from pkg_resources import resource_filename as pkgrf


class TestDataLoader:
    @pytest.fixture(autouse=True)
    def setup_class(self):
        self.fmriprep_dir = pkgrf('hypercoil', 'examples/ds-synth')
        self.models = [
            'rps',
            'wm + csf',
            'gsr',
            'acc<v=0.7, mask=WM+CSF>'
        ]
        self.tmask = 'and(1_[d1(rps) < 0.2])'
        self.ds = hypercoil.data.fMRIPrepDataset(
            self.fmriprep_dir, model=self.models, tmask=self.tmask)
        self.dl = hypercoil.data.dataset.ReferencedDataLoader(
            self.ds, batch_size=5, shuffle=False)

    def test_depth(self):
        assert len(self.ds.data_refs) == 10
        self.ds.set_depth(1)
        assert len(self.ds.data_refs) == 80
        self.ds.set_depth(0)

    def test_dl_depth_0(self):
        self.ds.set_depth(0)
        for sample in self.dl:
            break
        assert sample['images'].size() == torch.Size([5, 8, 4, 4, 4, 500])
        assert sample['rps'].size() == torch.Size([5, 8, 6, 500])
        assert sample['wm + csf'].size() == torch.Size([5, 8, 2, 500])
        assert sample['gsr'].size() == torch.Size([5, 8, 1, 500])
        assert sample['acc<v=0.7, mask=WM+CSF>'].size() == torch.Size(
            [5, 8, 4, 500])
        assert sample['t_r'].size() == torch.Size([5, 8, 1])
        assert sample['tmask'].size() == torch.Size([5, 8, 1, 500])

    def test_dl_depth_1(self):
        self.ds.set_depth(1)
        for sample in self.dl:
            break
        assert sample['images'].size() == torch.Size([5, 4, 4, 4, 500])
        assert sample['rps'].size() == torch.Size([5, 6, 500])
        assert sample['wm + csf'].size() == torch.Size([5, 2, 500])
        assert sample['gsr'].size() == torch.Size([5, 500])
        assert sample['acc<v=0.7, mask=WM+CSF>'].size() == torch.Size(
            [5, 3, 500])
        assert sample['t_r'].size() == torch.Size([5])
        assert sample['tmask'].size() == torch.Size([5, 500])
        assert sample['tmask'][0].sum() == 333
        self.ds.set_depth(0)
