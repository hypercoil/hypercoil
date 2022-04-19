# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Interpolation
~~~~~~~~~~~~~
Modules for performing interpolation.
"""
import torch
from ..functional import spectral_interpolate


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
