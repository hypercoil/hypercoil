# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Normed penalty
~~~~~~~~~~~~~~
Generalised module for applying a normed penalty to the weight parameter
of a module.
"""
from torch import diff, norm as pnorm
from torch.nn import Module
from functools import partial


class ReducingRegularisation(Module):
	def __init__(self, nu, reduction, reg):
		self.nu = nu
		self.reduction = reduction
		self.reg = reg

	def forward(self, weight):
		return self.nu * self.reduction(self.reg(weight))


class NormedRegularisation(ReducingRegularisation):
	def __init__(self, nu, p, reg):
		reduction = pnorm(p=p)
		super(NormedRegularisation, self).__init__(
			nu=nu, reduction=reduction, reg=reg
		)