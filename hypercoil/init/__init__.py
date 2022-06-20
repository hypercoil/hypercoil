# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Parameter initialisation schemes.

The ``init`` submodule contains functionality for initialising (neural
network) module parameters to reasonable defaults.

In the differentiable programming setting, initialisation provides a path for
integrating **domain expertise** into neural network models. The
initialisation schemes and routines provided here present a transparent
framework for incorporating the discipline's existing best practices as
starting points for a differentiable workflow.

For example, the routines here enable the
:doc:`linear atlas layer <api/hypercoil.nn.atlas.AtlasLinear>`
to be initialised as most available volumetric and surface-based neuroimaging
parcellations. Modifications can be applied easily to promote better learning
signals. For the case of the atlas layer, for instance, hard parcels can be
smoothed (in either Euclidean or
:doc:`spherical topology <api/hypercoil.functional.sphere>`
) or pre-transformed through a
:doc:`multi-logit <api/hypercoil.init.domain.MultiLogit>`
(softmax) domain transformation to change the properties of the gradients they
receive.

Also available are more general initialisation schemes for use cases where a
clean slate is desired as a starting point. For example, a random
:doc:`Dirichlet initialisation <api/hypercoil.init.dirichlet>`
, when combined with a
:doc:`multi-logit domain <api/hypercoil.init.domain.MultiLogit>`
, lends columns in a parcellation matrix the intuitive interpretation of
probability distributions over parcels.

Most initialisation scheme classes (eventually, all) can be combined with a
:doc:`domain mapper <api/hypercoil.init.domain>`.
If a domain mapper is used to initialise a compatible module's parameters, the
parameters are internally stored by the module as "pre-parameters" and then
transformed through the domain mapper before they interact with inputs.
Some schemes are paired with a domain mapper by default. For instance,
:doc:`Dirichlet initialisation <api/hypercoil.init.dirichlet>`
is by default paired with a
:doc:`multi-logit <api/hypercoil.init.domain.MultiLogit>`
domain to constrain Dirichlet-initialised weights to the probability simplex.

.. warning::
    Any and all APIs here are experimental and subject to change. Test
    coverage is inadequate and extreme discretion is warranted when using this
    functionality. Please report any bugs by opening an issue.
"""
from .domainbase import (
    Identity,
    Linear,
    Affine
)
from .domain import (
    Atanh,
    AmplitudeAtanh,
    Logit,
    MultiLogit,
    UnitNormSphere,
    AmplitudeMultiLogit
)
from .atlas import (
    DiscreteVolumetricAtlas,
    MultiVolumetricAtlas,
    MultifileVolumetricAtlas,
    CortexSubcortexCIfTIAtlas,
    DirichletInitVolumetricAtlas,
    DirichletInitSurfaceAtlas,
    AtlasInit
)
from .base import (
    from_distr_init_,
    constant_init_,
    identity_init_,
    DistributionInitialiser,
    ConstantInitialiser
)
from .deltaplus import (
    deltaplus_init_,
    DeltaPlusInit
)
from .dirichlet import (
    dirichlet_init_,
    DirichletInit
)
from .freqfilter import (
    FreqFilterSpec,
    freqfilter_init_,
    clamp_init_
)
from .iirfilter import (
    IIRFilterSpec
)
from .laplace import (
    laplace_init_,
    LaplaceInit
)
from .mpbl import (
    BipartiteLatticeInit
)
from .semidefinite import (
    tangency_init_,
    TangencyInit,
    SPDEuclideanMean,
    SPDHarmonicMean,
    SPDLogEuclideanMean,
    SPDGeometricMean
)
from .sylo import (
    sylo_init_
)
from .toeplitz import (
    toeplitz_init_,
    ToeplitzInit
)
