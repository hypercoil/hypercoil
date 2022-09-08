# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Modules that project data between the positive semidefinite cone proper
subspaces tangent to the cone.
"""
import jax
import equinox as eqx
from typing import Literal, Sequence, Tuple

from ..engine.axisutil import unfold_axes
from ..engine.paramutil import Tensor, _to_jax_array
from ..functional import (
    cone_project_spd, tangent_project_spd
)
from ..init.semidefinite import (
    mean_block_spd,
    TangencyInitialiser,
    _SemidefiniteMean,
)


class _TangentProject(eqx.Module):
    """
    Base class for modules that project symmetric matrix data between the
    positive semidefinite cone and proper subspaces tangent to the cone.
    """
    dest: Literal['tangent', 'cone']
    out_channels: int
    matrix_size: int
    recondition: float = 0.

    def __init__(
        self,
        out_channels: int,
        matrix_size: int,
        dest: Literal['tangent', 'cone'] = 'tangent',
        recondition: float = 0.,
        *,
        key: 'jax.random.PRNGKey' = None,
    ):
        self.dest = dest
        self.out_channels = out_channels
        self.matrix_size = matrix_size
        self.recondition = recondition

    def __call__(
        self,
        input: Tensor,
        weight: Tensor,
        dest: Literal['tangent', 'cone'] = None,
        *,
        key: 'jax.random.PRNGKey' = None,
    ):
        weight = _to_jax_array(weight)
        dest = dest or self.dest
        if self.out_channels > 1:
            input = input[..., None, :, :]
        if dest == 'tangent':
            out = tangent_project_spd(
                input, weight, self.recondition, key=key)
        elif dest == 'cone':
            out = cone_project_spd(
                input, weight, self.recondition, key=key)
        if self.out_channels > 1 and out.ndim > 4:
            out = unfold_axes(out, -4, -3)
        return out


class TangentProject(_TangentProject):
    r"""
    Tangent/cone projection with a learnable or fixed point of tangency.

    At initialisation, a data sample is required to set the point of tangency.
    In particular, the tangency point is initialised as a mean of the dataset,
    which can be the standard Euclidean mean or a measure of central tendency
    specifically derived for positive semidefinite matrices. Data transported
    through the module is projected from the positive semidefinite cone into a
    proper subspace tangent to the cone at the reference point which is the
    module weight. Given a tangency point :math:`\Omega`, each input
    :math:`\Theta` is projected as:

    :math:`\vec{\Theta} = \log \Omega^{-1/2} \Theta \Omega^{-1/2}`

    Alternatively, the module destination can be set to the semidefinite cone,
    in which case symmetric matrices are projected into the cone using the
    same reference point:

    :math:`\Theta = \Omega^{1/2} \exp \vec{\Theta} \Omega^{1/2}`

    From initialisation, the tangency point can be learned to optimise any
    differentiable loss.

    :Dimension: **Input :** :math:`(*, N, N)`
                    ``*`` denotes any number of preceding dimensions and N
                    denotes the size of each square symmetric matrix.
                **Output :** :math:`(*, C, N, N)`
                    `C` denotes the number of output channels (points of
                    tangency).

    Parameters
    ----------
    init_data : Tensor
        Data sample whose central tendency initialises the reference point of
        tangency.
    mean_specs : list(``_SemidefiniteMean`` object)
        Objects encoding a measure of central tendency in the positive
        semidefinite cone. Used to initialise the reference points of
        tangency. Selected from
        :doc:`means on the semidefinite cone <hypercoil.init.semidefinite>`.
    recondition : float in [0, 1]
        Conditioning factor :math:`\psi` to promote positive definiteness. If
        this is in (0, 1], the original input will be replaced with a convex
        combination of the input and an identity matrix.

        :math:`\hat{X} = (1 - \psi) X + \psi I`

        A suitable value can be used to ensure that all eigenvalues are
        positive and therefore guarantee that the matrix is in the domain.
    std : float
        Standard deviation of the positive semidefinite noise added to each
        channel of the weight matrix. This can be used to ensure that
        different channels initialised from the same mean receive different
        gradients and differentiate from one another.

    Attributes
    ----------
    weight : Tensor :math:`(*, C, N, N)`
        Reference point of tangency for the projection between the
        semidefinite cone and a tangent subspace.
    dest : ``'tangent'`` or ``'cone'``
        Target space/manifold of the projection operation.
    """
    weight: Tensor

    def __init__(
        self,
        out_channels: int,
        matrix_size: int,
        recondition: float = 0.,
        scale: float = 1.,
        dest: Literal['tangent', 'cone'] = 'tangent',
        *,
        key: 'jax.random.PRNGKey',
    ):
        super().__init__(
            out_channels=out_channels,
            matrix_size=matrix_size,
            dest=dest,
            recondition=recondition
        )
        weight = scale * jax.random.normal(
            key, (out_channels, matrix_size, matrix_size)
        )
        self.weight = weight @ weight.swapaxes(-1, -2)

    @classmethod
    def from_specs(
        cls,
        mean_specs: Sequence[_SemidefiniteMean],
        init_data: Tensor,
        recondition: float = 0.,
        dest: Literal['tangent', 'cone'] = 'tangent',
        std: float = 0.,
        *,
        key: 'jax.random.PRNGKey',
    ):
        m_key, i_key = jax.random.split(key)
        out_channels = len(mean_specs)
        matrix_size = init_data.shape[-1]
        model = cls(
            out_channels=out_channels,
            matrix_size=matrix_size,
            recondition=recondition,
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


class BatchTangentProject(_TangentProject):
    r"""
    Tangent/cone projection with a new tangency point computed for each batch.

    .. warning::
        Initialise this only using the ``from_specs`` class method.

    Data transported through the module is projected from the positive
    semidefinite cone into a proper subspace tangent to the cone at the
    reference point. Given a tangency point :math:`\Omega`, each input
    :math:`\Theta` is projected as:

    :math:`\vec{\Theta} = \log \Omega^{-1/2} \Theta \Omega^{-1/2}`

    Alternatively, the module destination can be set to the semidefinite cone,
    in which case symmetric matrices are projected into the cone using the
    same reference point:

    :math:`\Theta = \Omega^{1/2} \exp \vec{\Theta} \Omega^{1/2}`

    Here, the tangency point is computed as a convex combination of the
    previous tangency point and some measure of central tendency in the
    current data batch. The tangency point is *not* learnable. This module is
    almost definitely a bad idea, but it might somehow be helpful for
    regularisation, augmentation, or increasing the model's robustness to
    different views on the input data.

    The weight is updated ONLY during projection into tangent space. Given an
    inertial parameter :math:`\eta` and a measure of central tendency
    :math:`\bar{\Theta}`, the weight is updated as

    :math:`\Omega_t := \eta \Omega_{t-1} + (1 - \eta) \bar{\Theta}`

    :Dimension: **Input :** :math:`(*, N, N)`
                    ``*`` denotes any number of preceding dimensions and N
                    denotes the size of each square symmetric matrix.
                **Output :** :math:`(*, C, N, N)`
                    C denotes the number of output channels (points of
                    tangency).

    Parameters
    ----------
    mean_specs : list(_SemidefiniteMean object)
        Objects encoding a measure of central tendency in the positive
        semidefinite cone. Used to initialise the reference points of
        tangency. Selected from
        :doc:`means on the semidefinite cone <hypercoil.init.semidefinite>`.
    recondition : float in [0, 1]
        Conditioning factor :math:`\psi` to promote positive definiteness. If
        this is in (0, 1], the original input will be replaced with a convex
        combination of the input and an identity matrix.

        :math:`\hat{X} = (1 - \psi) X + \psi I`

        A suitable value can be used to ensure that all eigenvalues are
        positive and therefore guarantee that the matrix is in the domain.
    inertia : float in [0, 1]
        Parameter :math:`\eta` describing the relative weighting of the
        historical tangencypoint and the current batch mean. Zero inertia
        strictly uses the current batch mean. High inertia prevents the
        tangency point from skipping by heavily weighting the history.

    Attributes
    ----------
    weight : Tensor :math:`(*, C, N, N)`
        Current reference point of tangency :math:`\Omega` for the projection
        between the semidefinite cone and a tangent subspace.
    dest : ``'tangent'`` or ``'cone'``
        Target space/manifold of the projection operation.
    """
    inertia: float
    mean_specs: Sequence[_SemidefiniteMean]
    default_weight: Tensor

    def __init__(
        self,
        out_channels: int,
        matrix_size: int,
        mean_specs: Sequence[_SemidefiniteMean],
        recondition: float = 0.,
        scale: float = 1.,
        inertia: float = 0.,
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
            recondition=recondition
        )
        weight = scale * jax.random.normal(
            key, (out_channels, matrix_size, matrix_size)
        )
        self.default_weight = weight @ weight.swapaxes(-1, -2)

    @classmethod
    def from_specs(
        cls,
        mean_specs: Sequence[_SemidefiniteMean],
        init_data: Tensor,
        recondition: float = 0.,
        inertia: float = 0.,
        dest: Literal['tangent', 'cone'] = 'tangent',
        std: float = 0.,
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
            recondition=recondition,
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
            param_name='default_weight',
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
            #TODO: rather than a simple convex combination, use the module's
            # assigned mean.
            weight = self.inertia * weight + (1 - self.inertia) * input_weight
        out = super().__call__(input, weight, dest, key=key)
        return out, weight
