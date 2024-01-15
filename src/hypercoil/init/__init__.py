# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# isort: skip_file
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
) or constrained to the
:doc:`probability simplex <api/hypercoil.init.mapparam.ProbabilitySimplexParameter>`
(using a softmax domain transformation) to change the properties of the
gradients they receive.

Also available are more general initialisation schemes for use cases where a
clean slate is desired as a starting point. For example, a random
:doc:`Dirichlet initialisation <api/hypercoil.init.dirichlet>`
, when combined with a
:doc:`probability simplex projection <api/hypercoil.init.mapparam.ProbabilitySimplexParameter>`
, lends columns in a parcellation matrix the intuitive interpretation of
probability distributions over parcels.

Most initialisation scheme classes (eventually, all) can be combined with a
:doc:`parameter mapping <api/hypercoil.init.mapparam>`.
If a parameter mapping is used to initialise a compatible module's parameters,
the parameters are internally stored by the module as "original parameters"
and then transformed through the mapping before they interact with inputs.
Some schemes are paired with a mapping by default. For instance,
:doc:`Dirichlet initialisation <api/hypercoil.init.dirichlet>`
is by default paired with a
:doc:`probability simplex projection <api/hypercoil.init.domain.ProbabilitySimplexParameter>`
domain to constrain Dirichlet-initialised weights to always be valid
probability distributions.

.. warning::
    Any and all APIs here are experimental and subject to change. Test
    coverage is inadequate and extreme discretion is warranted when using this
    functionality. Please report any bugs by opening an issue.
"""
from .atlas import (
    DiscreteVolumetricAtlas,
    MultiVolumetricAtlas,
    MultifileVolumetricAtlas,
    CortexSubcortexCIfTIAtlas,
    DirichletInitVolumetricAtlas,
    DirichletInitSurfaceAtlas,
    AtlasInitialiser,
)
from .base import (
    from_distr_init,
    constant_init,
    identity_init,
    DistributionInitialiser,
    ConstantInitialiser,
    IdentityInitialiser,
    MappedInitialiser,
)
from .deltaplus import (
    deltaplus_init,
    DeltaPlusInitialiser,
)
from .dirichlet import (
    dirichlet_init,
    DirichletInitialiser,
)
from .freqfilter import (
    FreqFilterSpec,
    freqfilter_init,
    clamp_init
)
from .iirfilter import (
    IIRFilterSpec
)
from .laplace import (
    laplace_init,
    LaplaceInitialiser,
)
from .mapparam import (
    IdentityMappedParameter,
    AffineMappedParameter,
    TanhMappedParameter,
    AmplitudeTanhMappedParameter,
    MappedLogits,
    NormSphereParameter,
    ProbabilitySimplexParameter,
    AmplitudeProbabilitySimplexParameter,
    OrthogonalParameter,
    IsochoricParameter,
)
from .mpbl import (
    maximum_potential_bipartite_lattice,
)
from .semidefinite import (
    tangency_init,
    TangencyInitialiser,
    SPDEuclideanMean,
    SPDHarmonicMean,
    SPDLogEuclideanMean,
    SPDGeometricMean
)
from .sylo import (
    sylo_init,
    SyloInitialiser
)
from .toeplitz import (
    ToeplitzInitialiser
)
from .vmf import (
    VonMisesFisher
)
