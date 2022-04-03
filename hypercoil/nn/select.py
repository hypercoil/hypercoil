# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Model Selection
~~~~~~~~~~~~~~~
Modules supporting model selection, as for denoising/confound regression.
"""
import torch
from torch.nn import Module, Linear, Parameter
from ..functional.domain import Logit
from ..init.base import (
    DistributionInitialiser
)


class LinearCombinationSelector(Linear):
    r"""
    Model selection as a linear combination.

    Learn linear combinations of candidate vectors to produce a model. Thin
    wrapper around `torch.nn.Linear`.

    :Dimension: **Input :** :math:`(*, I, T)`
                    ``*`` denotes any number of preceding dimensions,
                    :math:`I` denotes number of candidate model vectors,
                    :math:`T` denotes number of time points or observations
                    per vector.
                **Output :** :math:`(*, O, T)`
                    :math:`O` denotes the final model dimension.

    Parameters
    ----------
    model_dim : int
        Dimension of the model to be learned.
    n_columns : int
        Number of input vectors to be combined linearly to form the model.

    Attributes
    ----------
    weight : tensor
        Tensor of shape :math:`(I, O)` `n_columns` x `model_dim`.
    """
    def __init__(self, model_dim, n_columns, dtype=None, device=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearCombinationSelector, self).__init__(
            in_features=n_columns,
            out_features=model_dim,
            bias=False,
            **factory_kwargs
        )

    def forward(self, x):
        return super().forward(x.transpose(-1, -2)).transpose(-1, -2)


class EliminationSelector(Module):
    r"""
    Model selection by elimination of variables.

    Begin with a full complement of model vectors, then eliminate them by
    placing an L1 penalty on the weight of this layer.

    The internal weights of this module are passed through a parameterised
    sigmoid function and then thresholded at 0. Any variables corresponding
    to a 0 weight are eliminated in the forward pass.

    :Dimension: **Input :** :math:`(*, I, T)`
                    ``*`` denotes any number of preceding dimensions,
                    :math:`I` denotes number of candidate model vectors,
                    :math:`T` denotes number of time points or observations
                    per vector.
                **Output :** :math:`(*, I, T)`

    Parameters
    ----------
    n_columns : int
        Number of candidate vectors for the model.
    infimum : float (default -1.5)
        Infimum of the thresholded sigmoid function, pre-thresholding. Note
        that an infimum closer to 0 results in a gentler slope close to the
        elimination threshold.
    supremum : float (default 2.5)
        Supremum of the thresholded sigmoid function.
    or_dim : int
        If this is greater than 1, then `or_dim` separate vectors are learned,
        and a variable is only eliminated if every one learns a 0 weight for
        that variable.
        During testing, we did not find a practical use for this. We didn't
        look very carefully, and it's possible that someone might find a use.
    and_dim : int
        If this is greater than 1, then `and_dim` separate vectors are
        learned, and a variable is eliminated if any one learns a 0 weight for
        that variable.
        If both `or_dim` and `and_dim` are set, then the selector first takes
        the union across the `or` dimension and then takes the intersection
        across the `and` dimension.
        During testing, we did not find a practical use for this. We didn't
        look very carefully, and it's possible that someone might find a use.
    init : callable
        Initialisation function for the layer weight. Defaults to values
        randomly sampled from Uniform(0, 1).
    """
    def __init__(self, n_columns, infimum=-1.5, supremum=2.5,
                 or_dim=1, and_dim=1, init=None, dtype=None, device=None):
        super(EliminationSelector, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.n_columns = n_columns
        self.or_dim = or_dim
        self.and_dim = and_dim
        self.preweight = Parameter(torch.empty(
            self.or_dim,
            self.and_dim,
            self.n_columns,
            **factory_kwargs
        ))
        scale = (supremum - infimum) / 2
        loc = supremum - scale
        #TODO: Can't we just use a ReLU domain? We might get more stable and
        # predictable behaviour that way as well. This module is currently
        # very sensitive to hyperparameters.
        self.domain = Logit(scale=scale, loc=loc)
        self.init = DistributionInitialiser(
            distr=torch.distributions.Uniform(0., 1.),
            domain=self.domain
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.init(self.preweight)

    @property
    def weight(self):
        w = self.domain.image(self.preweight)
        return torch.maximum(w, torch.tensor(
            0, dtype=self.preweight.dtype, device=self.preweight.device
        ))

    @property
    def postweight(self):
        return self.weight.sum(0).prod(0).view(-1, 1)

    def forward(self, x):
        return self.postweight * x
