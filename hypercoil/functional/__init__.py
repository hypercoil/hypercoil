# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Functional interfaces
"""
from .cov import (
    cov, corr, partialcorr, pairedcov, conditionalcov, conditionalcorr,
    precision, pairedcorr, covariance, correlation, corrcoef, pcorr, ccov,
    ccorr, partialcov
)
from .crosssim import (
    crosshair_similarity,
    crosshair_l1_similarity,
    crosshair_l2_similarity,
    crosshair_cosine_similarity
)
from .fourier import (
    product_filter,
    product_filtfilt
)
from .graph import (
    girvan_newman_null, modularity_matrix, relaxed_modularity
)
from .matrix import (
    invert_spd, expand_outer, spd, symmetric, toeplitz
)
from .noise import (
    DiagonalNoiseSource, SPSDNoiseSource,
    DiagonalDropoutSource, SPSDDropoutSource,
    LowRankNoiseSource, BandDropoutSource,
    UnstructuredNoiseSource, UnstructuredDropoutSource
)
from .polynomial import (
    polychan, polyconv2d
)
from .sylo import (
    sylo
)
from .symmap import (
    symmap, symexp, symlog, symsqrt
)
from .semidefinite import (
    tangent_project_spd, cone_project_spd, mean_euc_spd, mean_harm_spd,
    mean_logeuc_spd, mean_geom_spd
)