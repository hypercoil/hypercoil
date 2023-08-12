# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Modules supporting artefact modelling, as for denoising/confound regression.
"""
from __future__ import annotations
import math
from functools import partial
from typing import Callable, Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import equinox as eqx

from ..engine import Tensor, atleast_4d
from ..engine.paramutil import _to_jax_array
from ..functional import basischan, basisconv2d, tsconv2d
from ..init.mapparam import MappedLogits


class LinearRFNN(eqx.Module):
    """
    Model selection as a linear combination, with convolutional model
    augmentation.
    """

    model_dim: int
    basis_functions: Sequence[Callable[[Tensor], Tensor]]
    num_response_functions: int = 10
    response_duration: int = 9
    leak: float = 0.001
    weight: Dict[str, Tensor]

    def __init__(
        self,
        model_dim: int,
        num_columns: int,
        basis_functions: (
            Optional[Sequence[Callable[[Tensor], Tensor]]]
        ) = None,
        num_response_functions: int = 10,
        response_duration: int = 9,
        leak: float = 0.001,
        *,
        key: 'jax.random.PRNGKey',
    ):
        if basis_functions is None:
            basis_functions = (lambda x: x,)
        in_channels = len(basis_functions)
        num_columns = num_columns * (1 + num_response_functions)

        key_rf, key_lin, key_thresh = jax.random.split(key, 3)
        lim_lin = 1.0 / math.sqrt(num_columns)
        lim_rf = 1.0 / math.sqrt(in_channels * response_duration)
        self.weight = {
            'rf': jax.random.uniform(
                key_rf,
                shape=(
                    num_response_functions,
                    in_channels,
                    1,
                    response_duration,
                ),
                minval=-lim_rf,
                maxval=lim_rf,
            ),
            'lin': jax.random.uniform(
                key_lin,
                shape=(model_dim, num_columns),
                minval=-lim_lin,
                maxval=lim_lin,
            ),
            'thresh': 0.05
            * jax.random.normal(
                key_thresh,
                shape=(num_response_functions, 1, 1),
            ),
        }

        self.model_dim = model_dim
        self.num_response_functions = num_response_functions
        self.basis_functions = basis_functions
        self.leak = leak

    def __call__(
        self,
        x: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        weight = {k: _to_jax_array(v) for k, v in self.weight.items()}
        x = atleast_4d(x)

        # If we have no response functions, just select a linear combination
        if self.num_response_functions == 0:
            return weight['lin'] @ x

        # Step 1: Convolve with response functions
        rf_conv = basisconv2d(
            X=x,
            weight=weight['rf'],
            basis_functions=self.basis_functions,
            include_const=False,
            bias=None,
            padding=None,
        )

        # Step 2: Threshold
        rf_conv = rf_conv - weight['thresh']
        rf_conv = jax.nn.leaky_relu(rf_conv, negative_slope=self.leak)
        rf_conv = rf_conv + weight['thresh']
        n, c, v, t = rf_conv.shape

        # Step 3: Select the model as a linear combination
        all_functions = jnp.concatenate(
            (x, rf_conv.reshape(n, 1, c * v, t)),
            axis=-2,
        )
        return weight['lin'] @ all_functions


class QCPredict(eqx.Module):
    basis_functions: Sequence[Callable[[Tensor], Tensor]]
    leak: float = 0.05
    weight: Dict[str, Tensor]

    def __init__(
        self,
        num_columns: int,
        basis_functions: (
            Optional[Sequence[Callable[[Tensor], Tensor]]]
        ) = None,
        num_response_functions: int = 10,
        response_duration: int = 9,
        num_global_patterns: int = 10,
        global_pattern_duration: int = 9,
        final_filter_duration: int = 1,
        num_qc: int = 1,
        leak: float = 0.05,
        *,
        key: 'jax.random.PRNGKey',
    ):
        if basis_functions is None:
            basis_functions = (lambda x: x,)
        in_channels = len(basis_functions)

        (
            key_rf,
            key_global,
            key_final,
            key_thresh_rf,
            key_thresh_global,
            key_thresh_final,
        ) = jax.random.split(key, 6)
        lim_rf = 1.0 / math.sqrt(in_channels * response_duration)
        lim_global = 1.0 / math.sqrt(
            (num_response_functions + in_channels) * global_pattern_duration
        )
        lim_final = 1.0 / math.sqrt(
            (num_global_patterns + num_columns) * final_filter_duration
        )

        self.weight = {
            'rf': jax.random.uniform(
                key_rf,
                shape=(
                    num_response_functions,
                    in_channels,
                    1,
                    response_duration,
                ),
                minval=-lim_rf,
                maxval=lim_rf,
            ),
            'global': jax.random.uniform(
                key_global,
                shape=(
                    num_global_patterns,
                    num_response_functions + in_channels,
                    num_columns,
                    global_pattern_duration,
                ),
                minval=-lim_global,
                maxval=lim_global,
            ),
            'final': jax.random.uniform(
                key_final,
                shape=(
                    num_qc,
                    num_global_patterns + num_columns,
                    1,
                    final_filter_duration,
                ),
                minval=-lim_final,
                maxval=lim_final,
            ),
            'thresh_rf': 0.01
            * jax.random.normal(
                key_thresh_rf,
                shape=(num_response_functions, 1, 1),
            ),
            'thresh_global': 0.01
            * jax.random.normal(
                key_thresh_global,
                shape=(num_global_patterns, 1, 1),
            ),
            'thresh_final': 0.01
            * jax.random.normal(
                key_thresh_final,
                shape=(num_qc, 1, 1),
            ),
        }

        self.basis_functions = basis_functions
        self.leak = leak

    def conv_and_thresh(
        self,
        conv: Callable,
        x: Tensor,
        weight: Tensor,
        thresh: Tensor,
    ) -> Tensor:
        conv_out = conv(
            X=x,
            weight=weight,
            bias=None,
            padding=None,
        )
        return jax.nn.leaky_relu(conv_out - thresh, negative_slope=self.leak)

    def __call__(
        self,
        x: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        weight = {k: _to_jax_array(v) for k, v in self.weight.items()}
        rf_conv = self.conv_and_thresh(
            conv=partial(
                basisconv2d,
                basis_functions=self.basis_functions,
                include_const=False,
            ),
            x=x,
            weight=weight['rf'],
            thresh=weight['thresh_rf'],
        )
        n, c, v, t = rf_conv.shape
        augmented = jnp.concatenate(
            (
                basischan(
                    x,
                    basis_functions=self.basis_functions,
                ).reshape(n, -1, v, t),
                rf_conv,
            ),
            axis=1,
        )

        global_conv = self.conv_and_thresh(
            conv=tsconv2d,
            x=augmented,
            weight=weight['global'],
            thresh=weight['thresh_global'],
        )
        augmented = jnp.concatenate(
            (
                x.reshape(n, v, 1, t),
                global_conv,
            ),
            axis=1,
        )

        final_conv = self.conv_and_thresh(
            conv=tsconv2d,
            x=augmented,
            weight=weight['final'],
            thresh=weight['thresh_final'],
        )
        return final_conv


class LinearCombinationSelector(LinearRFNN):
    r"""
    Model selection as a linear combination.

    Learn linear combinations of candidate vectors to produce a model. Thin
    wrapper around :class:`LinearRFNN` without the convolutional layers for
    learning response functions.

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

    def __init__(
        self,
        model_dim: int,
        num_columns: int,
        *,
        key: 'jax.random.PRNGKey',
    ):
        super().__init__(
            model_dim=model_dim,
            num_columns=num_columns,
            num_response_functions=0,
            key=key,
        )


class EliminationSelector(eqx.Module):
    r"""
    Model selection by elimination of variables.

    Begin with a full complement of model vectors, then eliminate them by
    placing an L1 penalty on the weight of this layer.

    .. danger::

        Do not use this model! It once performed well as a fluke of
        initialisation, but testing it across multiple random seeds has shown
        that it is not a good model. It is included here as part of the
        synthetic data experiments, but should not be used in practice.

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
    num_columns: int
    or_dim: int = 1
    and_dim: int = 1
    weight: Tensor

    def __init__(
        self,
        num_columns: int,
        infimum: float = -1.5,
        supremum: float = 2.5,
        or_dim: int = 1,
        and_dim: int = 1,
        *,
        key: jax.random.PRNGKey,
    ):
        scale = (supremum - infimum) / 2
        loc = supremum - scale
        self.num_columns = num_columns
        self.or_dim = or_dim
        self.and_dim = and_dim
        self.weight = jax.random.uniform(
            key=key,
            shape=(or_dim, and_dim, num_columns),
            minval=0,
            maxval=1,
        )
        # TODO: Can't we just use a ReLU domain? We might get more stable and
        # predictable behaviour that way as well. This module is currently
        # very sensitive to hyperparameters.
        self.weight = MappedLogits(
            self,
            loc=loc,
            scale=scale,
        )

    def __call__(
        self,
        x: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        weight = _to_jax_array(self.weight)
        weight = jnp.maximum(weight, 0)
        weight = weight.sum(-3).prod(-2)
        weight = weight[..., None]
        return weight * x
