# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
"""
Neural network modules.
"""
from .activation import (
    CorrelationNorm,
    Isochor,
)
from .atlas import (
    AtlasLinear,
)
from .confound import (
    LinearRFNN,
    QCPredict,
    LinearCombinationSelector,
    EliminationSelector,
)
from .cov import(
    UnaryCovariance,
    UnaryCovarianceTW,
    UnaryCovarianceUW,
    BinaryCovariance,
    BinaryCovarianceTW,
    BinaryCovarianceUW,
)
from .freqfilter import (
    FrequencyDomainFilter,
)
from .interpolate import (
    SpectralInterpolate,
    LinearInterpolate,
    HybridInterpolate,
)
from .iirfilter import (
    IIRFilter,
)
from .recombinator import (
    Recombinator,
)
from .resid import (
    Residualise,
)
from .semidefinite import(
    TangentProject,
    BatchTangentProject,
)
from .sylo import (
    Sylo,
)
from .tsconv import (
    TimeSeriesConv2D,
    PolyConv2D,
    BasisConv2D,
)
from .window import (
    WindowAmplifier,
)
