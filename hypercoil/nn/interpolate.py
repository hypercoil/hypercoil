# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interpolation
~~~~~~~~~~~~~
Modules for performing interpolation.
"""
import torch
from ..functional import (
    spectral_interpolate,
    weighted_interpolate,
    hybrid_interpolate
)


class SpectralInterpolate(torch.nn.Module):
    def __init__(
        self,
        oversampling_frequency=8,
        maximum_frequency=1,
        sampling_period=1,
        thresh=0
    ):
        super().__init__()
        self.oversampling_frequency = oversampling_frequency
        self.maximum_frequency = maximum_frequency
        self.sampling_period = sampling_period
        self.thresh = thresh

    def forward(self, input, mask):
        return spectral_interpolate(
            data=input,
            tmask=mask,
            oversampling_frequency=self.oversampling_frequency,
            maximum_frequency=self.maximum_frequency,
            sampling_period=self.sampling_period,
            thresh=self.thresh
        )


class WeightedInterpolate(torch.nn.Module):
    def __init__(
        self,
        start_stage=1,
        max_stage=None,
        map_to_kernel=None
    ):
        super().__init__()
        self.start_stage = start_stage
        self.max_stage = max_stage
        self.map_to_kernel = map_to_kernel

    def forward(self, input, mask):
        return weighted_interpolate(
            data=input,
            mask=mask,
            start_stage=self.start_stage,
            max_stage=self.max_stage,
            map_to_kernel=self.map_to_kernel
        )


class HybridInterpolate(torch.nn.Module):
    def __init__(
        self,
        max_weighted_stage=3,
        map_to_kernel=None,
        oversampling_frequency=8,
        maximum_frequency=1,
        frequency_thresh=0.3
    ):
        super().__init__()
        self.max_weighted_stage = max_weighted_stage
        self.map_to_kernel = map_to_kernel
        self.oversampling_frequency = oversampling_frequency
        self.maximum_frequency = maximum_frequency
        self.frequency_thresh = frequency_thresh

    def forward(self, input, mask):
        return hybrid_interpolate(
            data=input,
            mask=mask,
            max_weighted_stage=self.max_weighted_stage,
            map_to_kernel=self.map_to_kernel,
            oversampling_frequency=self.oversampling_frequency,
            maximum_frequency=self.maximum_frequency,
            frequency_thresh=self.frequency_thresh
        )
