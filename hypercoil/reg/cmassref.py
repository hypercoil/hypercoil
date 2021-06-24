# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Reference proximity
~~~~~~~~~~~~~~~~~~~
Regularisations using centre-of-mass proximity to a reference.
"""
from functools import partial
from .norm import NormedRegularisation
from ..functional.cmass import cmass


def cmass_reference_distance(weight, refs, axes=None, na_rm=False):
	cm = cmass(weight, axes=axes, na_rm=na_rm)
	return cm - refs


class CentreToReference(NormedRegularisation):
	def __init__(self, refs, nu=1, axes=None, na_rm=False, norm=2):
		reg = partial(
			cmass_reference_distance,
			refs=refs,
			axes=axes,
			na_rm=na_rm
		)
        super(SymmetricBimodal, self).__init__(nu=nu, p=norm, reg=reg)
