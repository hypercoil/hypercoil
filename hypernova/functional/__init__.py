# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Functional interfaces
"""
from .cov import (
    cov, corr, partialcorr, pairedcov, conditionalcov, conditionalcorr,
    precision, invert_spd, covariance, correlation, corrcoef, pcorr, ccov,
    ccorr
)
from .semidefinite import (
    tangent_project_spd, cone_project_spd, mean_euc_spd, mean_harm_spd,
    mean_logeuc_spd
)
from .symmap import (
    symmap, symexp, symlog, symsqrt
)
