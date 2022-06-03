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

All loss objects additionally inherit
:doc:`sentry <hypercoil.engine.sentry>` functionality, enabling them to send
and receive information about events from elsewhere and respond accordingly.
For instance, this can facilitate archival of each loss value over the course
of training to inform hyperparameter tuning. It can also be used to update
multipliers over the course of training by listening for epoch changes.
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
    CentroidAnchor,
    HemisphericTether
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
from .jsdiv import (
    JSDivergence,
    SoftmaxJSDivergence
)
from .modularity import (
    ModularityLoss
)
from .mvkurtosis import (
    MultivariateKurtosis
)
from .norm import (
    NormedLoss,
    UnilateralNormedLoss
)
from .scheme import (
    LossScheme
)
from .secondmoment import (
    SecondMoment,
    SecondMomentCentred
)
from .smoothness import (
    SmoothnessPenalty
)
from .symbimodal import (
    SymmetricBimodal,
    SymmetricBimodalNorm
)
