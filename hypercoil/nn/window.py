# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Window amplification
~~~~~~~~~~~~~~~~~~~~
Rebatching or channel amplification through data windowing.
"""
import torch
from torch.nn import Module


class WindowAmplifier(Module):
    def __init__(self, window_size, augmentation_factor=1,
                 nonoverlapping=False, augmentation_axis=0):
        super().__init__()
        self.window_size = window_size
        self.augmentation_factor = augmentation_factor
        self.augmentation_axis = augmentation_axis
        self.nonoverlapping = nonoverlapping

    def _get_data_dim(self, tensor):
        return tensor.size(-1)

    def _select_windows(self, avail):
        windows = []
        aug = self.augmentation_factor
        size = self.window_size
        minim = 0
        for w in range(self.augmentation_factor):
            if self.nonoverlapping:
                ##TODO: this simple scheme can be very biased to selecting
                # later time points. We should come up with something more
                # sophisticated later.
                size = self.window_size * aug
                aug -= 1
            try:
                start = torch.randint(
                    low=minim, high=(avail - size + 1), size=(1,)).item()
            except RuntimeError:
                raise ValueError(
                    f'Impossible to partition data of size {avail} '
                    f'into {self.augmentation_factor} nonoverlapping '
                    f'windows of size {self.window_size}'
                )
            windows += [(start, start + self.window_size)]
            if self.nonoverlapping:
                minim = start + self.window_size
        return windows

    def forward(self, data):
        if isinstance(data, torch.Tensor):
            data = [data]
        avail = self._get_data_dim(data[0])
        windows = self._select_windows(avail)
        out = []
        for tensor in data:
            try:
                out += [torch.cat([
                    tensor[..., start:end] for start, end in windows
                ], dim=self.augmentation_axis)]
            except RuntimeError:
                dims = ' and '.join([str(t.size(-1)) for t in data])
                raise RuntimeError(
                    f'All input tensors should have the same final axis '
                    f'dimension. Detected input shapes {dims}.')
        return out
