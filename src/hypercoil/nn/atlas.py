# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Modules that map voxelwise signals to labelwise signals.
"""
from __future__ import annotations
from collections import OrderedDict
from typing import Callable, Dict, Literal, Optional, Tuple, Type

import jax
import jax.numpy as jnp
import equinox as eqx
from numpyro.distributions import Dirichlet

from ..engine import NestedDocParse, PyTree, Tensor
from ..engine.paramutil import _to_jax_array
from ..functional.linear import compartmentalised_linear
from ..init.atlas import AtlasInitialiser, BaseAtlas
from ..init.mapparam import MappedParameter


def document_atlas_lin_init(f: Callable) -> Callable:
    atlas_lin_init_param_spec = r"""
    normalisation : ``'mean'``, ``'absmean'``, ``'zscore'``, ``'psc'``, or None (default ``'mean'``)
        Strategy for normalising across voxels and generating a representative
        time series for each label.

        * None or ``sum``: No normalisation, i.e., use the weighted sum over
          voxel time series.
        * ``mean``: Compute the weighted mean over voxel time series.
        * ``absmean``: Compute the weighted mean over voxel time series,
          treating any negative voxel weights as though they were positive.
        * ``zscore``: Transform the sum of time series such that its temporal
          mean is 0 and its temporal standard deviation is 1.
        * ``psc``: Transform the time series such that its value indicates the
          percent signal change from the mean.
    forward_mode : ``'map'`` or ``'project'``
        Strategy for extracting regional time series from parcels.

        * ``'map'``: Simple linear map. Given a compartment atlas
          :math:`A \in \mathbb{R}^{(L \times V)}`
          and a vertex-wise or voxel-wise input time series
          :math:`T_{in} \in \mathbb{R}^{(V \times T)}`, returns

          :math:`T_{out} = A T_{in}`.
        * ``'project'``: Projection using a linear least-squares fit. Given a
          compartment atlas
          :math:`A \in \mathbb{R}^{(L \times V)}`
          and a vertex-wise or voxel-wise input time series
          :math:`T_{in} \in \mathbb{R}^{(V \times T)}`, returns

          .. math::

            \begin{aligned}
            T_{out} &= \min_{X \in \mathbb{R}^{(L \times T)}} \| A^\intercal X - T_{in} \|_F

            &= \left(A A^\intercal\right)^{-1} A T_{in}
            \end{aligned}
    concatenate : bool, optional (default=True)
        Whether to concatenate the output time series across compartments."""

    fmt = NestedDocParse(
        atlas_lin_init_param_spec=atlas_lin_init_param_spec,
    )
    f.__doc__ = f.__doc__.format_map(fmt)
    return f


@document_atlas_lin_init
class AtlasLinear(eqx.Module):
    r"""
    Time series extraction from an atlas via a linear map.

    :Dimension: **Input :** :math:`(N, *, V, T)`
                    N denotes batch size, `*` denotes any number of intervening
                    dimensions, V denotes total number of voxels or spatial
                    locations, T denotes number of time points or observations.
                **Output :** :math:`(N, *, L, T)`
                    L denotes number of labels in the provided atlas.

    .. note::

        To initialise the atlas linear module from a pre-defined atlas, use
        the class method :meth:`from_atlas` or the
        :class:`hypercoil.init.atlas.AtlasInitialiser` class after defining
        the atlas as a :class:`hypercoil.init.atlas.BaseAtlas` instance.

        If the module is initialised withouth an atlas, the atlas linear
        module will be initialised from a Dirichlet distribution with
        concentration :math:`50` for each label.

    Parameters
    ----------
    n_locations : Dict[str, int]
        Number of locations (e.g., voxels or vertices) in each compartment.
    n_labels : Dict[str, int]
        Number of labels in each compartment.
    limits : Dict[str, Tuple[int, int]], optional (default=None)
        Limits of each compartment. The first element of the tuple denotes the
        lower limit and the second element denotes the size, i.e., the number
        of locations in the compartment -- *not* the upper limit. If None, the
        limits are set to the default values, which are defined using the
        cumulative sum of the number of locations in each compartment.
    decoder : Optional[Dict[str, Tensor]], optional (default=None)
        Decoder for labels in each compartment. The decoder is an
        integer-valued tensor that defines the map from row numbers to label
        numbers. If None, the decoder corresponds to the identity map -- i.e.,
        the row numbers are the same as the label numbers.\
    {atlas_lin_init_param_spec}
    """
    weight: Dict[str, Tensor]
    limits: Dict[str, Tuple[int, int]]
    decoder: Optional[Dict[str, Tensor]]
    normalisation: (
        Optional[Literal["mean", "absmean", "zscore", "psc"]]
    ) = None
    forward_mode: Literal["map", "project"] = "map"
    concatenate: bool = True

    def __init__(
        self,
        n_locations: Dict[str, int],
        n_labels: Dict[str, int],
        limits: Dict[str, Tuple[int, int]] = None,
        decoder: Optional[Dict[str, Tensor]] = None,
        normalisation: (
            Optional[Literal["mean", "absmean", "zscore", "psc"]]
        ) = "mean",
        forward_mode: Literal["map", "project"] = "map",
        concatenate: bool = True,
        *,
        key: "jax.random.PRNGKey",
    ):
        compartments = set(n_locations.keys())
        if compartments != set(n_labels.keys()):
            raise ValueError(
                "n_locations and n_labels must have the same keys"
            )
        keys = jax.random.split(key, len(compartments))

        self.weight = {
            c: (
                Dirichlet(
                    concentration=(50 * jnp.ones(n_labels[c])),
                )
                .sample(
                    key=k,
                    sample_shape=(n_locations[c],),
                )
                .swapaxes(-2, -1)
                if n_labels[c] > 1
                else jnp.ones((1, n_locations[c]))
            )
            for c, k in zip(compartments, keys)
        }

        if limits is None:
            limits = self.set_default_limits(n_locations)

        self.limits = limits
        self.decoder = decoder
        self.normalisation = normalisation
        self.forward_mode = forward_mode
        self.concatenate = concatenate

    @staticmethod
    def set_default_limits(
        n_locations: Dict[str, int],
    ) -> Dict[str, Tuple[float, float]]:
        limits = {}
        index = 0
        for c in n_locations.keys():
            limits[c] = (index, n_locations[c])
            index += n_locations[c]
        return limits

    @classmethod
    @document_atlas_lin_init
    def from_atlas(
        cls,
        atlas: BaseAtlas,
        normalisation: (
            Optional[Literal["mean", "absmean", "zscore", "psc"]]
        ) = "mean",
        forward_mode: Literal["map", "project"] = "map",
        concatenate: bool = True,
        *,
        mapper: Optional[Type[MappedParameter]] = None,
        normalise: bool = False,
        max_bin: int = 10000,
        spherical_scale: float = 1.0,
        truncate: Optional[float] = None,
        kernel_sigma: Optional[float] = None,
        noise_sigma: Optional[float] = None,
        where: str = "weight",
        key: "jax.random.PRNGKey",
        **params,
    ) -> PyTree:
        """
        Initialise the atlas module from an instance of a ``BaseAtlas``
        subclass, perhaps representing *a priori* knowledge about the
        functional organisation of the brain.

        Parameters
        ----------
        atlas : Atlas object
            A neuroimaging atlas, implemented as an instance of a subclass of
            :doc:`BaseAtlas <hypercoil.init.atlas.BaseAtlas>`.
            This initialises the atlas labels from which representative time
            series are extracted.\
        {atlas_lin_init_param_spec}
        """
        m_key, i_key = jax.random.split(key, 2)
        n_locations = {c: o.size for c, o in atlas.compartments.items()}
        n_labels = {
            c: atlas.maps[c].shape[-2] if atlas.maps[c].shape != (0,) else 0
            for c in atlas.compartments
        }
        limits = {
            c: (o.slice_index, o.slice_size)
            for c, o in atlas.compartments.items()
        }
        model = cls(
            n_locations=n_locations,
            n_labels=n_labels,
            limits=limits,
            decoder=atlas.decoder,  # TODO: we're maintaining a reference to
            #      the atlas here. Is this a problem?
            normalisation=normalisation,
            forward_mode=forward_mode,
            concatenate=concatenate,
            key=m_key,
        )
        model = AtlasInitialiser.init(
            model,
            atlas=atlas,
            mapper=mapper,
            normalise=normalise,
            max_bin=max_bin,
            spherical_scale=spherical_scale,
            truncate=truncate,
            kernel_sigma=kernel_sigma,
            noise_sigma=noise_sigma,
            where=where,
            key=i_key,
            **params,
        )
        return model

    def __call__(
        self,
        input: Tensor,
        normalisation: (
            Optional[Literal["mean", "absmean", "zscore", "psc"]]
        ) = None,
        forward_mode: Optional[Literal["map", "project"]] = None,
        concatenate: Optional[bool] = None,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ) -> Tensor:
        if normalisation is None:
            normalisation = self.normalisation
        if forward_mode is None:
            forward_mode = self.forward_mode
        if concatenate is None:
            concatenate = self.concatenate
        weight = OrderedDict(
            ((k, _to_jax_array(v)) for k, v in self.weight.items())
        )
        return compartmentalised_linear(
            input=input,
            weight=weight,
            limits=self.limits,
            decoder=self.decoder,
            normalisation=normalisation,
            forward_mode=forward_mode,
            concatenate=concatenate,
        )


# TODO: experimental "accumuline" functionality. We can revisit this after we
#      figure out a decent way to do it in JAX, where accumulating operands
#      over multiple forward passes is made significantly more difficult by
#      the fact that we can't easily maintain state between calls.

# def atlas_backward(atlas, grad_output, *grad_compartments):
#     """
#     Backward pass through the atlas layer. Any domain mapper gradients are not
#     included.

#     During the forward pass, compartment-specific local Jacobian matrices must
#     be cached as an iterable whose elements follow the same ordering as the
#     iteration through atlas preweights (parameters).
#     """
#     grad_output = grad_output.squeeze(0)
#     ret = []
#     offset = 0
#     i = 0
#     for name in atlas.preweight.keys():
#         code = atlas.atlas.decoder[name]
#         if not atlas.decode:
#             n_labels = len(code)
#             code = torch.arange(
#                 offset,
#                 offset + n_labels,
#                 dtype=torch.long,
#                 device=grad_output.device
#             )
#             offset += n_labels
#         grad_out_compartment = grad_output[code] @ grad_compartments[i]
#         ret += [grad_out_compartment]
#         # indexing seems a little dangerous
#         i += 1
#     return tuple(ret)


# def atlas_gradient(atlas, input, *args, **kwargs):
#     """
#     Local derivative across the atlas layer, of the output with respect to the
#     weights. For each compartment-specific weight, the local derivative is the
#     transpose of that compartment's time series.
#     """
#     compartment_grads = ModelArgument()
#     for name in atlas.preweight.keys():
#         compartment_ts = atlas.select_compartment(name, input)
#         compartment_grads[name] = compartment_ts.transpose(-1, -2)
#     return compartment_grads


# def atlas_accfn(atlas, input, acc, argmap=None, out=[], terminate=False):
#     fwd = AccumulatingFunction.apply
#     def bwd(grad_output, *grad_compartments):
#         return atlas_backward(atlas, grad_output, *grad_compartments)
#     if argmap is None: argmap = lambda x: ModelArgument(input=x)
#     params = [atlas.weight[name] for name in atlas.preweight.keys()]
#     return fwd(
#         acc,
#         bwd,
#         argmap,
#         input,
#         out,
#         terminate,
#         *params
#     )


# class AtlasAccumuline(Accumuline):
#     """
#     :class:`AtlasLinear` layer with
#     :doc:`Accumuline <hypercoil.engine.accumulate.Accumuline>`
#     functionality for
#     :doc:`local gradient accumulation and rebatching <hypercoil.engine.accumulate>`.

#     .. warning::
#         This is untested functionality and it will not work.
#     """
#     def __init__(
#         self,
#         atlas,
#         origin,
#         throughput,
#         batch_size,
#         image_key=None,
#         reduction=None,
#         argmap=None,
#         influx=None,
#         efflux=None,
#         lines=None,
#         transmit_filters=None,
#         receive_filters=None,
#         skip_local=False,
#         nonlocal_argmap=None,
#     ):
#         reduction = reduction or 'mean'
#         image_key = image_key or 'images'
#         argmap = argmap or (lambda x: ModelArgument(input=x))
#         gradient = partial(atlas_gradient, atlas=atlas)
#         accfn = partial(atlas_accfn, atlas=atlas, argmap=argmap)
#         local_argmap = self.argmap
#         influx = influx or (
#             lambda arg: UnpackingModelArgument(input=arg.images))
#         super().__init__(
#             model=atlas,
#             accfn=accfn,
#             gradient=gradient,
#             origin=origin,
#             retain_dims=(-1, -2),
#             throughput=throughput,
#             batch_size=batch_size,
#             reduction=reduction,
#             params=None,
#             influx=influx,
#             efflux=efflux,
#             lines=lines,
#             transmit_filters=transmit_filters,
#             receive_filters=receive_filters,
#             skip_local=skip_local,
#             local_argmap=local_argmap,
#             nonlocal_argmap=nonlocal_argmap,
#         )
#         self.coors = {}
#         self.masks = {}
#         self.ref = atlas.atlas
#         self.image_key = image_key
#         for name in self.model.preweight.keys():
#             compartment = self.ref.compartments[name]
#             self.masks[name] = compartment[atlas.mask]
#             self.coors[name] = self.ref.coors[self.masks[name]].t()

#     def argmap(self, input, atlas):
#         images = input[self.image_key]
#         # Note that we require a second forward pass to get our arg.
#         # Set skip_local if you don't need it.
#         output = self.model(images)
#         inputs = {
#             name : apply_mask(images, mask, -2)
#             for name, mask in self.masks.items()
#         }
#         inputs = ModelArgument(**inputs)
#         weights = ModelArgument(**self.model.weight)
#         preweights = ModelArgument(**self.model.preweight)
#         coors = ModelArgument(**self.coors)
#         return ModelArgument(
#             input=input,
#             ts=inputs,
#             output=output,
#             preweight=preweights,
#             weight=weights,
#             coor=coors
#         )

#     def forward(self):
#         return super().forward(atlas=self.model)
