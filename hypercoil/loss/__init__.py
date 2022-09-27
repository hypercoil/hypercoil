# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Loss functions and regularisations.

The ``loss`` submodule is a collection of implementations of differentiable
functions that map from arbitrary inputs to scalar-valued outputs. These
scalar outputs can provide a starting point for a backward pass through a
differentiable program model. Functionality is provided for various measures
of interest to functional brain mapping and other contexts.

Helper wrappers allow packaging of multiple loss objectives in a single call.
Each wrapped objective can be selectively applied to a subset of input tensors
using
:doc:`LossApply <hypercoil.loss.base.LossApply>`,
:doc:`LossArgument <hypercoil.engine.argument.ModelArgument>`, and
:doc:`LossScheme <hypercoil.loss.base.LossScheme>`
functionality.
"""
from .functional import (
    identity,
    zero,
    difference,
    constraint_violation,
    unilateral_loss,
    hinge_loss,
    smoothness,
    bimodal_symmetric,
    det_gram,
    log_det_gram,
    entropy,
    entropy_logit,
    kl_divergence,
    kl_divergence_logit,
    js_divergence,
    js_divergence_logit,
    bregman_divergence,
    bregman_divergence_logit,
    equilibrium,
    equilibrium_logit,
    second_moment,
    second_moment_centred,
    auto_tol,
    batch_corr,
    qcfc,
    reference_tether,
    interhemispheric_tether,
    compactness,
    dispersion,
    multivariate_kurtosis,
    connectopy,
    modularity,
)
from .nn import (
    Loss,
    ParameterisedLoss,
    MSELoss,
    NormedLoss,
    ConstraintViolationLoss,
    UnilateralLoss,
    HingeLoss,
    SmoothnessLoss,
    BimodalSymmetricLoss,
    GramDeterminantLoss,
    GramLogDeterminantLoss,
    EntropyLoss,
    EntropyLogitLoss,
    KLDivergenceLoss,
    KLDivergenceLogitLoss,
    JSDivergenceLoss,
    JSDivergenceLogitLoss,
    BregmanDivergenceLoss,
    BregmanDivergenceLogitLoss,
    EquilibriumLoss,
    EquilibriumLogitLoss,
    SecondMomentLoss,
    SecondMomentCentredLoss,
    BatchCorrelationLoss,
    QCFCLoss,
    ReferenceTetherLoss,
    InterhemisphericTetherLoss,
    CompactnessLoss,
    DispersionLoss,
    MultivariateKurtosis,
    ConnectopyLoss,
    ModularityLoss,
)
from .scalarise import (
    sum_scalarise,
    mean_scalarise,
    meansq_scalarise,
    max_scalarise,
    norm_scalarise,
    vnorm_scalarise,
    wmean_scalarise,
    selfwmean_scalarise,
)
from .scheme import (
    LossApply,
    LossScheme,
    LossReturn,
    LossArgument,
    UnpackingLossArgument,
)
