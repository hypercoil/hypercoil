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
    def __init__(self, model_dim, n_columns):
        super(LinearCombinationSelector, self).__init__(
            in_features=n_columns,
            out_features=model_dim,
            bias=False
        )

    def forward(self, x):
        return super().forward(x.transpose(-1, -2)).transpose(-1, -2)


class EliminationSelector(Module):
    def __init__(self, n_columns, infimum=-1.5, supremum=2.5,
                 or_dim=1, and_dim=1, init=None):
        super(EliminationSelector, self).__init__()
        self.n_columns = n_columns
        self.or_dim = or_dim
        self.and_dim = and_dim
        self.preweight = Parameter(torch.Tensor(
            self.or_dim,
            self.and_dim,
            self.n_columns
        ))
        scale = (supremum - infimum) / 2
        loc = supremum - scale
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
        return torch.maximum(w, torch.tensor(0))

    @property
    def postweight(self):
        return self.weight.sum(0).prod(0).view(-1, 1)

    def forward(self, x):
        return self.postweight * x
