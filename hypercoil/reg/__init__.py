# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Initialisation
"""
from .norm import (
    ReducingRegularisation,
    NormedRegularisation,
    UnilateralNormedRegularisation
)
from .scheme import (
    RegularisationScheme
)
from .smoothness import (
    SmoothnessPenalty
)
from .symbimodal import (
    SymmetricBimodal
)
