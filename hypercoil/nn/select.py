# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Model Selection
~~~~~~~~~~~~~~~
Modules supporting model selection, as for denoising/confound regression.
"""
import torch
from torch.nn import Module, Linear, Parameter, ParameterDict
from functools import partial
from ..functional.domain import Logit
from ..functional import basischan, basisconv2d, tsconv2d, threshold
from ..init.dirichlet import DirichletInit
from ..init.base import (
    DistributionInitialiser
)


class ResponseFunctionLinearSelector(Module):
    """
    Model selection as a linear combination, with convolutional model
    augmentation.
    """
    def __init__(
        self,
        model_dim,
        n_columns,
        basis_functions=None,
        n_response_functions=10,
        response_function_len=9,
        init_lin=None,
        init_conv=None
    ):
        super().__init__()
        n_columns = n_columns * (1 + n_response_functions)
        if basis_functions is None:
            basis_functions = [lambda x : x]
        if init_lin is None:
            init_lin = lambda x: torch.nn.init.xavier_uniform_(x)
        if init_conv is None:
            init_conv = lambda x: torch.nn.init.kaiming_uniform_(
                x, nonlinearity='relu')
        self.weight = ParameterDict({
            'rf' : Parameter(torch.empty(
                n_response_functions,
                len(basis_functions),
                1,
                response_function_len)),
            'lin' : Parameter(torch.empty(
                model_dim, n_columns)),
            'thresh' : Parameter(torch.empty(
                n_response_functions, 1, 1))
        })
        self.init_lin = init_lin
        self.init_conv = init_conv
        self.basis_functions = basis_functions
        self.reset_parameters()

    def reset_parameters(self):
        self.init_conv(self.weight['rf'])
        self.init_lin(self.weight['lin'])
        torch.nn.init.normal_(self.weight['thresh'], 0.05)

    def forward(self, x):
        rf_conv = basisconv2d(
            X=x,
            weight=self.weight['rf'],
            basis_functions=self.basis_functions,
            include_const=False,
            bias=None,
            padding=None)
        rf_conv = rf_conv - self.weight['thresh']
        rf_conv = torch.relu(rf_conv)
        rf_conv = rf_conv + self.weight['thresh']
        n, c, h, w = rf_conv.shape
        all_functions = torch.cat((
            x,
            rf_conv.view(n, c * h, w)
        ), axis=-2)
        return self.weight['lin'] @ all_functions


class QCPredict(Module):
    def __init__(
        self,
        n_ts,
        basis_functions=None,
        n_response_functions=10,
        response_function_len=9,
        n_global_patterns=10,
        global_pattern_len=9,
        final_filter_len=1,
        n_qc=1,
        init_rf=None,
        init_global=None,
        init_final=None
    ):
        super().__init__()
        default_init = lambda x: torch.nn.init.kaiming_uniform_(
            x, nonlinearity='relu')
        self.weight = ParameterDict({
            'rf' : Parameter(torch.empty(
                n_response_functions,
                len(basis_functions),
                1,
                response_function_len)),
            'global' : Parameter(torch.empty(
                n_global_patterns,
                n_response_functions + len(basis_functions),
                n_ts,
                global_pattern_len)),
            'final' : Parameter(torch.empty(
                n_qc,
                n_global_patterns + n_ts,
                1,
                final_filter_len)),
            'thresh_rf' : Parameter(torch.empty(
                n_response_functions, 1, 1)),
            'thresh_global' : Parameter(torch.empty(
                n_global_patterns, 1, 1)),
            'thresh_final' : Parameter(torch.empty(
                n_qc, 1, 1))
        })
        self.init_rf = init_rf or  default_init
        self.init_global = init_global or default_init
        self.init_final = init_final or default_init
        self.basis_functions = basis_functions or [lambda x : x]
        self.reset_parameters()

    def reset_parameters(self):
        self.init_rf(self.weight['rf'])
        self.init_global(self.weight['global'])
        self.init_final(self.weight['final'])
        torch.nn.init.normal_(self.weight['thresh_rf'], 0.01)
        torch.nn.init.normal_(self.weight['thresh_global'], 0.01)
        torch.nn.init.normal_(self.weight['thresh_final'], 0.01)

    def conv_and_thresh(self, conv, x, weight, thresh):
        conv_out = conv(
            X=x,
            weight=weight,
            bias=None,
            padding=None)
        return threshold(conv_out, threshold=thresh)

    def forward(self, x):
        rf_conv = self.conv_and_thresh(
            conv=partial(basisconv2d,
                         basis_functions=self.basis_functions,
                         include_const=False),
            x=x,
            weight=self.weight['rf'],
            thresh=self.weight['thresh_rf'])
        n, c, h, w = rf_conv.shape
        augmented = torch.cat((
            basischan(x, basis_functions=self.basis_functions).view(n, -1, h, w),
            rf_conv
        ), axis=1)

        global_conv = self.conv_and_thresh(
            conv=tsconv2d,
            x=augmented,
            weight=self.weight['global'],
            thresh=self.weight['thresh_global'])
        augmented = torch.cat((
            x.view(n, h, 1, w),
            global_conv
        ), axis=1)

        final_conv = self.conv_and_thresh(
            conv=tsconv2d,
            x=augmented,
            weight=self.weight['final'],
            thresh=self.weight['thresh_final'])
        return final_conv


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
