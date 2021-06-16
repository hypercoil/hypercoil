# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Positive semidefinite cone
~~~~~~~~~~~~~~~~~~~~~~~~~~
Modules that project data between the positive semidefinite cone proper
subspaces tangent to the cone.
"""
import torch
from torch.nn import Module, Parameter
from ..functional import (
    cone_project_spd, tangent_project_spd
)
from ..init.semidefinite import mean_block_spd, TangencyInit


class _TangentProject(Module):
    """
    Base class for modules that project symmetric matrix data between the
    positive semidefinite cone and proper subspaces tangent to the cone.
    """
    def __init__(self, mean_specs=None, recondition=0):
        super(_TangentProject, self).__init__()
        self.dest = 'tangent'
        self.mean_specs = mean_specs or [SPDEuclideanMean()]
        self.out_channels = len(self.mean_specs)
        self.recondition = recondition

    def extra_repr(self):
        s = ',\n'.join([f'(mean) {spec.__repr__()}'
                         for spec in self.mean_specs])
        if self.recondition != 0:
            s += f',\npsi={self.recondition}'
        if self.out_channels > 1:
            s += f',\nout_channels={self.out_channels}'
        return s

    def forward(self, input, dest=None):
        if self.out_channels > 1:
            input = input.unsqueeze(-3)
        dest = dest or self.dest
        if dest == 'tangent':
            return tangent_project_spd(input, self.weight, self.recondition)
        elif dest == 'cone':
            return cone_project_spd(input, self.weight, self.recondition)


class TangentProject(_TangentProject):
    r"""
    Tangent/cone projection with a learnable or fixed point of tangency.

    At initialisation, a data sample is required to set the point of tangency.
    In particular, the tangency point is initialised as a mean of the dataset,
    which can be the standard Euclidean mean or a measure of central tendency
    specifically created for positive semidefinite matrices. Data transported
    through the module is projected from the positive semidefinite cone into a
    proper subspace tangent to the cone at the reference point which is the
    module weight. Given a tangency point :math:`\Omega`, each input
    :math:`\Theta` is projected as:

    :math:`\bar{\Theta} = \log_M \Omega^{-1/2} \Theta \Omega^{-1/2}`

    Alternatively, the module destination can be set to the semidefinite cone,
    in which case symmetric matrices are projected into the cone using the same
    reference point:

    :math:`\bar{\Theta} = \Omega^{1/2} \exp_M \Theta \Omega^{1/2}`

    From initialisation, the tangency point can be learned to optimise any
    differentiable loss.

    Dimension
    ---------
    - Input: :math:`(*, N, N)`
      `*` denotes any number of preceding dimensions and N denotes the size of
      each square symmetric matrix.
    - Output: :math:`(*, C, N, N)`
      C denotes the number of output channels (points of tangency).

    Parameters
    ----------
    init_data : Tensor
        Data sample whose central tendency initialises the reference point of
        tangency.
    mean_specs : list(_SemidefiniteMean object)
        Objects encoding a measure of central tendency in the positive
        semidefinite cone. Used to initialise the reference points of tangency.
    recondition : float in [0, 1]
        Conditioning factor :math:`psi` to promote positive definiteness. If
        this is in (0, 1], the original input will be replaced with a convex
        combination of the input and an identity matrix.

        :math:`\hat{X} = (1 - \psi) X + \psi I`

        A suitable value can be used to ensure that all eigenvalues are
        positive and therefore guarantee that the matrix is in the domain.
    std : float
        Standard deviation of the positive semidefinite noise added to each
        channel of the weight matrix. This can be used to ensure that different
        channels initialised from the same mean receive different gradients
        and differentiate from one another.

    Attributes
    ----------
    weight : Tensor :math:`(*, C, N, N)`
        Reference point of tangency for the projection between the semidefinite
        cone and a tangent subspace.
    dest : `tangent` or `cone`
        Target space/manifold of the projection operation.
    """
    def __init__(self, init_data, mean_specs=None, recondition=0, std=0):
        super(TangentProject, self).__init__(mean_specs, recondition)
        if self.out_channels > 1:
            self.weight = Parameter(torch.Tensor(
                *init_data.size()[1:-2],
                self.out_channels,
                init_data.size(-2),
                init_data.size(-1),
            ))
        else:
            self.weight = Parameter(torch.Tensor(
                *init_data.size()[1:]
            ))
        self.init = TangencyInit(
            self.mean_specs, init_data, std=std
        )
        self.reset_parameters(init_data, std)

    def reset_parameters(self, init_data, std=0):
        self.init(self.weight)


class BatchTangentProject(_TangentProject):
    r"""
    Tangent/cone projection with a new tangency point computed for each batch.

    Data transported through the module is projected from the positive
    semidefinite cone into a proper subspace tangent to the cone at the
    reference point. Given a tangency point :math:`\Omega`, each input
    :math:`\Theta` is projected as:

    :math:`\bar{\Theta} = \log_M \Omega^{-1/2} \Theta \Omega^{-1/2}`

    Alternatively, the module destination can be set to the semidefinite cone,
    in which case symmetric matrices are projected into the cone using the same
    reference point:

    :math:`\bar{\Theta} = \Omega^{1/2} \exp_M \Theta \Omega^{1/2}`

    Here, the tangency point is computed as a convex combination of the
    previous tangency point and some measure of central tendency in the current
    data batch. The tangency point is *not* learnable. This module is almost
    definitely a bad idea, but it might somehow be helpful for regularisation,
    augmentation, or increasing the model's robustness to different views on
    the input data.

    The weight is updated ONLY during projection into tangent space.

    Dimension
    ---------
    - Input: :math:`(*, N, N)`
      `*` denotes any number of preceding dimensions and N denotes the size of
      each square symmetric matrix.
    - Output: :math:`(*, C, N, N)`
      C denotes the number of output channels (points of tangency).

    Parameters
    ----------
    mean_specs : list(_SemidefiniteMean object)
        Objects encoding a measure of central tendency in the positive
        semidefinite cone. Used to initialise the reference points of tangency.
    recondition : float in [0, 1]
        Conditioning factor :math:`psi` to promote positive definiteness. If
        this is in (0, 1], the original input will be replaced with a convex
        combination of the input and an identity matrix.

        :math:`\hat{X} = (1 - \psi) X + \psi I`

        A suitable value can be used to ensure that all eigenvalues are
        positive and therefore guarantee that the matrix is in the domain.
    inertia : float in [0, 1]
        Parameter describing the relative weighting of the historical tangency
        point and the current batch mean. Zero inertia strictly uses the
        current batch mean. High inertia prevents the tangency point from
        skipping by heavily weighting the history.

    Attributes
    ----------
    weight : Tensor :math:`(*, C, N, N)`
        Current reference point of tangency for the projection between the
        semidefinite cone and a tangent subspace.
    dest : `tangent` or `cone`
        Target space/manifold of the projection operation.
    """
    def __init__(self, mean_specs=None, recondition=0, inertia=0):
        super(BatchTangentProject, self).__init__(mean_specs, recondition)
        self.inertia = inertia
        self.weight = None

    def forward(self, input, dest=None):
        if dest != 'cone':
            weight = mean_block_spd(self.mean_specs, input)
            if self.weight is None:
                self.weight = weight.detach()
            self.weight = (
                self.inertia * self.weight + (1 - self.inertia) * weight
            ).detach()
        elif self.weight is None:
            raise ValueError('Undefined weight: project into tangent space '
                             'first to initialise.')
        out = super(BatchTangentProject, self).forward(input, dest)
        return out
