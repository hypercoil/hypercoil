# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Modules that project data between the positive semidefinite cone and proper
subspaces tangent to the cone.
"""
from __future__ import annotations
from typing import Callable, Literal, Optional, Sequence, Tuple

import jax
import equinox as eqx

from ..engine.axisutil import unfold_axes
from ..engine.docutil import NestedDocParse
from ..engine.paramutil import Tensor, _to_jax_array
from ..functional.semidefinite import (
    cone_project_spd,
    document_semidefinite_mean,
    tangent_project_spd,
)
from ..init.semidefinite import (
    TangencyInitialiser,
    _SemidefiniteMean,
    mean_block_spd,
)


# TODO: Enable either convex combinations or eigenspace reconditioning
#       operations to ensure nonsingular and nondegenerate matrices.
# TODO: Reconcile all of the many docstrings we have for symmetric positive
#       definite matrices. The point of docstring transformation is to make
#       maintenance easier, but scattering the docstrings across the codebase
#       is not the way to do it.


def document_semidefinite_projection_module(f: Callable) -> Callable:
    long_description = r"""
    Data transported through the module is projected from the positive
    semidefinite cone into a proper subspace tangent to the cone at the
    reference point which is the module weight. Given a tangency point
    :math:`\Omega`, each input :math:`\Theta` is projected as:

    :math:`\vec{\Theta} = \log \Omega^{-1/2} \Theta \Omega^{-1/2}`

    Alternatively, the module destination can be set to the semidefinite cone,
    in which case symmetric matrices are projected into the cone using the same
    reference point:

    :math:`\Theta = \Omega^{1/2} \exp \vec{\Theta} \Omega^{1/2}`"""

    tangent_projection_dim = r"""
    :Dimension: **Input :** :math:`(*, N, N)`
                    ``*`` denotes any number of preceding dimensions and N
                    denotes the size of each square symmetric matrix.
                **Output :** :math:`(*, C, N, N)`
                    C denotes the number of output channels (points of
                    tangency)."""

    tangent_projection_mean_specs_spec = """
    mean_specs : list(``_SemidefiniteMean`` object)
        Objects encoding a measure of central tendency in the positive
        semidefinite cone. Used to initialise the reference points of
        tangency. Selected from
        :doc:`means on the semidefinite cone <hypercoil.init.semidefinite>`."""

    tangent_projection_dest_spec = """
    dest : ``'tangent'`` or ``'cone'``
        Target space/manifold of the projection operation."""

    fmt = NestedDocParse(
        tangent_projection_long_description=long_description,
        tangent_projection_dim=tangent_projection_dim,
        tangent_projection_mean_specs_spec=tangent_projection_mean_specs_spec,
        tangent_projection_dest_spec=tangent_projection_dest_spec,
    )
    f.__doc__ = f.__doc__.format_map(fmt)
    return f


def document_semidefinite_init_from_spec(f: Callable) -> Callable:
    f.__doc__ = """
    Initialise the reference point of tangency from a data sample.

    See :doc:`means on the semidefinite cone <hypercoil.init.semidefinite>`
    for a list of available measures of central tendency, and
    :doc:`the initialiser class <hypercoil.init.semidefinite.TangencyInitialiser>`
    for more details on the initialisation procedure.
    """
    return f


class _TangentProject(eqx.Module):
    """
    Base class for modules that project symmetric matrix data between the
    positive semidefinite cone and proper subspaces tangent to the cone.
    """

    dest: Literal['tangent', 'cone']
    out_channels: int
    matrix_size: int
    psi: float = 0.0

    def __init__(
        self,
        out_channels: int,
        matrix_size: int,
        dest: Literal['tangent', 'cone'] = 'tangent',
        psi: float = 0.0,
        *,
        key: 'jax.random.PRNGKey' = None,
    ):
        self.dest = dest
        self.out_channels = out_channels
        self.matrix_size = matrix_size
        self.psi = psi

    def __call__(
        self,
        input: Tensor,
        weight: Tensor,
        dest: Literal['tangent', 'cone'] = None,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ):
        weight = _to_jax_array(weight)
        dest = dest or self.dest
        if self.out_channels > 1:
            input = input[..., None, :, :]
        if dest == 'tangent':
            out = tangent_project_spd(
                input,
                reference=weight,
                psi=self.psi,
                recondition='convexcombination',
                key=key,
            )
        elif dest == 'cone':
            out = cone_project_spd(
                input,
                reference=weight,
                psi=self.psi,
                recondition='convexcombination',
                key=key,
            )
        if self.out_channels > 1 and out.ndim > 4:
            out = unfold_axes(out, -4, -3)
        return out


@document_semidefinite_mean
@document_semidefinite_projection_module
class TangentProject(_TangentProject):
    """
    Tangent/cone projection with a learnable or fixed point of tangency.

    At initialisation, a data sample is required to set the point of tangency.
    In particular, the tangency point is initialised as a mean of the dataset,
    which can be the standard Euclidean mean or a measure of central tendency
    specifically derived for positive semidefinite matrices. \
    {tangent_projection_long_description}

    From initialisation, the tangency point can be learned to optimise any
    differentiable loss.
    \
    {tangent_projection_dim}

    Parameters
    ----------
    init_data : Tensor
        Data sample whose central tendency initialises the reference point of
        tangency.\
    {tangent_projection_mean_specs_spec}\
    {semidefinite_mean_psi_spec}
    scale : float
        Scaling factor for the initialisation of the reference point of
        tangency.\
    {tangent_projection_dest_spec}

    Attributes
    ----------
    weight : Tensor :math:`(*, C, N, N)`
        Reference point of tangency for the projection between the
        semidefinite cone and a tangent subspace.
    """

    weight: Tensor

    def __init__(
        self,
        out_channels: int,
        matrix_size: int,
        psi: float = 0.0,
        scale: float = 1.0,
        dest: Literal['tangent', 'cone'] = 'tangent',
        *,
        key: 'jax.random.PRNGKey',
    ):
        super().__init__(
            out_channels=out_channels,
            matrix_size=matrix_size,
            dest=dest,
            psi=psi,
        )
        weight = scale * jax.random.normal(
            key, (out_channels, matrix_size, matrix_size)
        )
        self.weight = weight @ weight.swapaxes(-1, -2)

    @classmethod
    @document_semidefinite_init_from_spec
    def from_specs(
        cls,
        mean_specs: Sequence[_SemidefiniteMean],
        init_data: Tensor,
        psi: float = 0.0,
        dest: Literal['tangent', 'cone'] = 'tangent',
        std: float = 0.0,
        *,
        key: 'jax.random.PRNGKey',
    ):
        """
        Initialise the reference point of tangency from a data sample.

        See :doc:`means on the semidefinite cone <hypercoil.init.semidefinite>`
        for a list of available measures of central tendency, and
        :doc:`the initialiser class <hypercoil.init.semidefinite.TangencyInitialiser>`
        for more details on the initialisation procedure.
        """
        m_key, i_key = jax.random.split(key)
        out_channels = len(mean_specs)
        matrix_size = init_data.shape[-1]
        model = cls(
            out_channels=out_channels,
            matrix_size=matrix_size,
            psi=psi,
            dest=dest,
            key=m_key,
        )
        return TangencyInitialiser.init(
            model,
            init_data=init_data,
            mean_specs=mean_specs,
            std=std,
            key=i_key,
        )

    def __call__(
        self,
        input: Tensor,
        dest: Literal['tangent', 'cone'] = None,
        *,
        key: 'jax.random.PRNGKey' = None,
    ):
        return super().__call__(input, self.weight, dest, key=key)


@document_semidefinite_mean
@document_semidefinite_projection_module
class BatchTangentProject(_TangentProject):
    """
    Tangent/cone projection with a new tangency point computed for each batch.

    .. warning::
        Initialise this only using the ``from_specs`` class method.
    \
    {tangent_projection_long_description}

    Here, the tangency point is computed as a convex combination of the
    previous tangency point and some measure of central tendency in the
    current data batch. The tangency point is *not* learnable. This module is
    almost definitely a bad idea, but it might somehow be helpful for
    regularisation, augmentation, or increasing the model's robustness to
    different views on the input data.

    The weight is updated *only* during projection into tangent space. Given
    an inertial parameter :math:`\eta` and a measure of central tendency
    :math:`\bar{\Theta}`, the weight is updated as

    :math:`\Omega_t := \eta \Omega_{t-1} + (1 - \eta) \bar{\Theta}`
    \
    {tangent_projection_dim}

    Parameters
    ----------\
    {tangent_projection_mean_specs_spec}\
    {semidefinite_mean_psi_spec}
    inertia : float in [0, 1]
        Parameter :math:`\eta` describing the relative weighting of the
        historical tangencypoint and the current batch mean. Zero inertia
        strictly uses the current batch mean. High inertia prevents the
        tangency point from skipping by heavily weighting the history.\
    {tangent_projection_dest_spec}

    Attributes
    ----------
    weight : Tensor :math:`(*, C, N, N)`
        Current reference point of tangency :math:`\Omega` for the projection
        between the semidefinite cone and a tangent subspace.
    """

    inertia: float
    mean_specs: Sequence[_SemidefiniteMean]
    default_weight: Tensor

    def __init__(
        self,
        out_channels: int,
        matrix_size: int,
        mean_specs: Sequence[_SemidefiniteMean],
        psi: float = 0.0,
        scale: float = 1.0,
        inertia: float = 0.0,
        dest: Literal['tangent', 'cone'] = 'tangent',
        *,
        key: 'jax.random.PRNGKey',
    ):
        self.inertia = inertia
        self.mean_specs = mean_specs
        super().__init__(
            out_channels=out_channels,
            matrix_size=matrix_size,
            dest=dest,
            psi=psi,
        )
        weight = scale * jax.random.normal(
            key, (out_channels, matrix_size, matrix_size)
        )
        self.default_weight = weight @ weight.swapaxes(-1, -2)

    @classmethod
    @document_semidefinite_init_from_spec
    def from_specs(
        cls,
        mean_specs: Sequence[_SemidefiniteMean],
        init_data: Tensor,
        psi: float = 0.0,
        inertia: float = 0.0,
        dest: Literal['tangent', 'cone'] = 'tangent',
        std: float = 0.0,
        *,
        key: 'jax.random.PRNGKey',
    ):
        m_key, i_key = jax.random.split(key)
        out_channels = len(mean_specs)
        matrix_size = init_data.shape[-1]
        model = cls(
            out_channels=out_channels,
            matrix_size=matrix_size,
            mean_specs=mean_specs,
            psi=psi,
            inertia=inertia,
            dest=dest,
            key=m_key,
        )
        return TangencyInitialiser.init(
            model,
            init_data=init_data,
            mean_specs=mean_specs,
            std=std,
            key=i_key,
            where='default_weight',
        )

    def __call__(
        self,
        input: Tensor,
        weight: Tensor = None,
        dest: Literal['tangent', 'cone'] = None,
        *,
        key: 'jax.random.PRNGKey' = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Project a batch of matrices into the destination manifold, updating
        the weight (reference point for projection) if the destination is a
        tangent subspace.

        If no ``dest`` parameter is provided, then the destination manifold
        will be set to the module's internal default destination attribute.
        """
        dest = dest or self.dest
        if weight is None:
            weight = self.default_weight
        if dest == 'tangent':
            input_weight = mean_block_spd(self.mean_specs, input)
            # TODO: rather than a simple convex combination, use the module's
            # assigned mean.
            weight = self.inertia * weight + (1 - self.inertia) * input_weight
        out = super().__call__(input, weight, dest, key=key)
        return out, weight
