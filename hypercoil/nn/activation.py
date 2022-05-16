# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Activation
~~~~~~~~~~
Activation function modules.
"""
import torch
from torch.nn import Module
from ..functional.activation import corrnorm


class CorrelationNorm(Module):
	def __init__(self, factor=None, grad_path='both'):
		super().__init__()
		self.factor = factor
		self.grad_path = grad_path

	def forward(self, input, factor=None):
		if factor is None:
			factor = self.factor
		return corrnorm(
			input=input,
			factor=factor,
			gradpath=self.grad_path
		)
