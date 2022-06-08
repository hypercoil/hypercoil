# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Neural network modules.
"""
from .atlas import AtlasLinear, AtlasAccumuline
from .cov import(
    UnaryCovariance, UnaryCovarianceTW, UnaryCovarianceUW,
    BinaryCovariance, BinaryCovarianceTW, BinaryCovarianceUW
)
from .freqfilter import (
    FrequencyDomainFilter
)
from .interpolate import (
    SpectralInterpolate,
    WeightedInterpolate,
    HybridInterpolate
)
from .iirfilter import (
    IIRFilter,
    IIRFiltFilt
)
from .polyconv import (
    PolyConv2D
)
from .recombinator import (
    Recombinator
)
from .resid import (
    Residualise
)
from .select import (
    ResponseFunctionLinearSelector,
    QCPredict,
    LinearCombinationSelector,
    EliminationSelector
)
from .semidefinite import(
    TangentProject, BatchTangentProject
)
from .spdnoise import (
    SPDNoise
)
from .svm import (
    SVM
)
from .sylo import (
    Sylo
)
from .window import (
    WindowAmplifier
)
