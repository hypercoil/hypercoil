# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Initialisation
"""
from .base import (
    LossApply,
    ReducingLoss,
    LossArgument,
    UnpackingLossArgument
)
from .batchcorr import (
    BatchCorrelation,
    QCFC
)
from .cmass import (
    Compactness,
    CentroidAnchor
)
from .determinant import (
    Determinant,
    LogDet,
    DetCorr,
    LogDetCorr
)
from .dispersion import (
    VectorDispersion
)
from .entropy import (
    Entropy,
    SoftmaxEntropy
)
from .equilibrium import (
    Equilibrium,
    SoftmaxEquilibrium
)
from .hinge import (
    HingeLoss
)
from .modularity import (
    ModularityLoss
)
from .norm import (
    NormedLoss,
    UnilateralNormedLoss
)
from .scheme import (
    LossScheme
)
from .secondmoment import (
    SecondMoment
)
from .smoothness import (
    SmoothnessPenalty
)
from .symbimodal import (
    SymmetricBimodal,
    SymmetricBimodalNorm
)
