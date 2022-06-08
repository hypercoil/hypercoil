# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Modules that map voxelwise signals to labelwise signals.
"""
import torch
from operator import mul
from functools import reduce, partial
from collections import OrderedDict
from torch.nn import Module, Parameter, ParameterDict
from torch.distributions import Bernoulli
from ..engine.accumulate import Accumuline, AccumulatingFunction
from ..engine.argument import ModelArgument, UnpackingModelArgument
from ..init.domain import Identity
from ..engine.noise import UnstructuredDropoutSource
from ..functional.utils import apply_mask
from ..init.atlas import AtlasInit


#TODO: add projection forward mode
class AtlasLinear(Module):
    r"""
    Time series extraction from an atlas via a linear map.

    Dimension
    ---------
    - Input: :math:`(N, *, V, T)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      V denotes total number of voxels or spatial locations, T denotes number
      of time points or observations.
    - Output: :math:`(N, *, L, T)`
      L denotes number of labels in the provided atlas.

    Parameters
    ----------
    atlas : Atlas object
        A neuroimaging atlas, implemented as an instance of a subclass of
        :doc:`BaseAtlas <hypercoil.init.atlas.BaseAtlas>`.
        This initialises the atlas labels from which representative time
        series are extracted.
    kernel_sigma : float (default None)
        If this is a float, then a Gaussian smoothing kernel with the
        specified width is applied to each label at initialisation.
    noise_sigma : float (default None)
        If this is a float, then Gaussian noise with the specified standard
        deviation is added to the label at initialisation.
    mask_input : bool (default False)
        Indicates that each input contains non-atlas locations and should be
        masked before time series extraction. If True, then the boolean tensor
        stored in the ``mask`` field of the ``atlas`` input is used to subset
        each input.
    spatial_dropout : float in [0, 1) (default 0)
        Probability of dropout for each voxel. If this is nonzero, then during
        training each voxel's weight has some probability of being set to zero,
        thus discounting the voxel from the time series estimate.
        Conjecturally, this can perhaps promote learning a weight that is
        robust to the influence of any single voxel.
    min_voxels : positive int (default 1)
        Minimum number of voxels that each region must contain after dropout.
        If a random dropout results in fewer remaining voxels, then another
        random dropout will be sampled until the criterion is satisfied. Has no
        effect if ``spatial_dropout`` is zero.
    domain : Domain object (default ``Identity``)
        A domain mapper from
        :doc:`hypercoil.init.domain <hypercoil.init.domain>`, used to specify
        the domain of the atlas weights. An
        :doc:`Identity <hypercoil.init.domainbase.Identity>`
        object yields the raw atlas weights, while an
        :doc:`Atanh <hypercoil.init.domain.Atanh>` object constrains weights
        to ``(-a, a)``, and a :doc:`Logit <hypercoil.init.domain.Logit>`
        object constrains weights to ``(0, a)`` by transforming the raw
        weights through a tanh or sigmoid function, respectively. A
        :doc:`MultiLogit <hypercoil.init.domain.MultiLogit>` domain mapper
        lends the atlas an intuitive interpretation as a probabilistic
        parcellation. Using an appropriate domain can ensure that weights are
        nonnegative and that they do not grow explosively.
    reduce : ``'mean'``, ``'absmean'``, ``'zscore'``, ``'psc'``, or ``'sum'`` (default ``'mean'``)
        Strategy for reducing across voxels and generating a representative
        time series for each label.

        * ``sum``: Weighted sum over voxel time series.
        * ``mean``: Compute the weighted mean over voxel time series.
        * ``absmean``: Compute the weighted mean over voxel time series,
          treating any negative voxel weights as though they were positive.
        * ``zscore``: Transform the sum of time series such that its temporal
          mean is 0 and its temporal standard deviation is 1.
        * ``psc``: Transform the time series such that its value indicates the
          percent signal change from the mean. (**untested**))

    Attributes
    ----------
    preweight : Tensor :math:`(L, V)`
        Atlas map in the module's domain. L denotes the number of labels, and
        V denotes the number of voxels. Identical to ``weight`` if the domain
        is ``Identity``.
    weight : Tensor :math:`(L, V)`
        Representation of the atlas as a linear map from voxels to labels,
        applied independently to each time point in each input image.
    postweight : Tensor :math:`(L, V)`
        Atlas map after application of spatial dropout or noise. Spatial
        dropout has a chance of randomly removing each voxel from
        consideration when extracting each time series. Spatial dropout
        is applied only during training. Identical to ``weight`` if there is
        no spatial dropout.
    mask : Tensor :math:`(V)`
        Boolean-valued tensor indicating the voxels that should be included as
        inputs to the atlas transformation.
    coors : Tensor :math:`(V, D)`
        Spatial coordinates of each location in the atlas.
    """
    def __init__(
        self,
        atlas,
        mask_input=False,
        normalise=True,
        compartments=True,
        concatenate=True,
        decode=False,
        kernel_sigma=None,
        noise_sigma=None,
        max_bin=10000,
        spherical_scale=1,
        truncate=None,
        domain=None,
        spatial_dropout=0,
        min_voxels=1,
        reduction='mean',
        dtype=None,
        device=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AtlasLinear, self).__init__()

        self.atlas = atlas
        self.mask = self.atlas.mask
        self.mask_input = mask_input
        self.reduction = reduction
        self.concatenate = concatenate
        self.decode = decode
        self.init = AtlasInit(
            self.atlas,
            normalise=False,
            max_bin=max_bin,
            spherical_scale=spherical_scale,
            kernel_sigma=kernel_sigma,
            noise_sigma=noise_sigma,
            truncate=truncate,
            domain=domain
        )
        self.domain = self.init.domain

        if compartments:
            self.preweight = ParameterDict({
                c : Parameter(torch.empty_like(self.atlas.maps[c],
                                               **factory_kwargs))
                for c in atlas.compartments.keys()
                if reduce(mul, self.atlas.maps[c].shape) != 0
            })
        else:
            self.preweight = ParameterDict({
                '_all' : Parameter(torch.empty_like(self.atlas.maps['_all'],
                                                    **factory_kwargs))
            })
        self.coors = self.atlas.coors
        self._configure_spatial_dropout(spatial_dropout, min_voxels)
        self.reset_parameters()

    def reset_parameters(self):
       self.init(tensor=self.preweight)

    def _configure_spatial_dropout(self, dropout_rate, min_voxels):
        if dropout_rate > 0:
            self.dropout = UnstructuredDropoutSource(
                distr=Bernoulli(1 - dropout_rate),
                sample_axes=[-1]
            )
        else:
            self.dropout = None
        self.min_voxels = min_voxels

    @property
    def weight(self):
        return OrderedDict([
            (k, self.domain.image(v))
            for k, v in self.preweight.items()
        ])

    @property
    def postweight(self):
        if self.dropout is not None:
            weight = OrderedDict()
            for k, v in self.weight.items():
                sufficient_voxels = False
                while not sufficient_voxels:
                    n_voxels = (v > 0).sum(-1)
                    weight[k] = self.dropout(v)
                    n_voxels = (weight[k] > 0).sum(-1)
                    if torch.all(n_voxels >= self.min_voxels):
                        sufficient_voxels = True
            return weight
        return self.weight

    def reduce(self, input, weight):
        out = weight @ input
        if self.reduction == 'mean':
            normfact = weight.sum(-1, keepdim=True)
            return out / normfact
        elif self.reduction == 'absmean':
            normfact = weight.abs().sum(-1, keepdim=True)
            return out / normfact
        elif self.reduction == 'zscore':
            out -= out.mean(-1, keepdim=True)
            out /= out.std(-1, keepdim=True)
            return out
        elif self.reduction == 'psc':
            mean = out.mean(-1, keepdim=True)
            return 100 * (out - mean) / mean
        return out

    def _conform_mask_to_input(self, mask, input):
        while mask.dim() < input.dim() - 1:
            mask = mask.unsqueeze(0)
        return mask.expand(input.shape[:-1])


    def apply_mask(self, input):
        shape = input.size()
        mask = self._conform_mask_to_input(mask=self.mask, input=input)
        input = input[mask]
        input = input.view(*shape[:-2], -1 , shape[-1])
        return input

    def select_compartment(self, compartment, input):
        shape = input.size()
        mask = self.atlas.compartments[compartment]
        mask = mask[self.mask]
        mask = self._conform_mask_to_input(mask=mask, input=input)
        input = input[mask].view(*shape[:-2], -1 , shape[-1])
        return input

    def concatenate_and_decode(self, input):
        if self.decode:
            k = list(self.weight.keys())[0]
            shape = (
                *input[k].shape[:-2],
                len(self.atlas.decoder['_all']),
                input[k].shape[-1]
            )
            out = torch.empty(
                *shape,
                dtype=input[k].dtype,
                device=input[k].device
            )
            for compartment, tensor in input.items():
                out[..., (self.atlas.decoder[compartment] - 1), :] = tensor
        #TODO: Let's use an OrderedDict to be safe.
        elif self.concatenate:
            out = torch.cat([v for v in input.values()], -2)
        return out

    def forward(self, input):
        if self.mask_input:
            input = self.apply_mask(input)
        out = OrderedDict()
        for k in self.preweight.keys():
            #print(compartment, self.atlas.compartments)
            v = self.postweight[k]
            if v.shape == (0,):
                continue
            compartment = self.select_compartment(k, input)
            out[k] = self.reduce(compartment, v)
        out = self.concatenate_and_decode(out)
        return out

    def __repr__(self):
        s = f'{type(self).__name__}(atlas={self.atlas}, '
        s += f'mask={self.mask_input}, reduce={self.reduction})'
        return s


def atlas_backward(atlas, grad_output, *grad_compartments):
    """
    Backward pass through the atlas layer. Any domain mapper gradients are not
    included.

    During the forward pass, compartment-specific local Jacobian matrices must
    be cached as an iterable whose elements follow the same ordering as the
    iteration through atlas preweights (parameters).
    """
    grad_output = grad_output.squeeze(0)
    ret = []
    offset = 0
    i = 0
    for name in atlas.preweight.keys():
        code = atlas.atlas.decoder[name]
        if not atlas.decode:
            n_labels = len(code)
            code = torch.arange(
                offset,
                offset + n_labels,
                dtype=torch.long,
                device=grad_output.device
            )
            offset += n_labels
        grad_out_compartment = grad_output[code] @ grad_compartments[i]
        ret += [grad_out_compartment]
        # indexing seems a little dangerous
        i += 1
    return tuple(ret)


def atlas_gradient(atlas, input, *args, **kwargs):
    """
    Local derivative across the atlas layer, of the output with respect to the
    weights. For each compartment-specific weight, the local derivative is the
    transpose of that compartment's time series.
    """
    compartment_grads = ModelArgument()
    for name in atlas.preweight.keys():
        compartment_ts = atlas.select_compartment(name, input)
        compartment_grads[name] = compartment_ts.transpose(-1, -2)
    return compartment_grads


def atlas_accfn(atlas, input, acc, argmap=None, out=[], terminate=False):
    fwd = AccumulatingFunction.apply
    def bwd(grad_output, *grad_compartments):
        return atlas_backward(atlas, grad_output, *grad_compartments)
    if argmap is None: argmap = lambda x: ModelArgument(input=x)
    params = [atlas.weight[name] for name in atlas.preweight.keys()]
    return fwd(
        acc,
        bwd,
        argmap,
        input,
        out,
        terminate,
        *params
    )


class AtlasAccumuline(Accumuline):
    """
    :class:`AtlasLinear` layer with
    :doc:`Accumuline <hypercoil.engine.accumulate.Accumuline>`
    functionality for
    :doc:`local gradient accumulation and rebatching <hypercoil.engine.accumulate>`.

    .. warning::
        This is untested functionality and it will not work.
    """
    def __init__(
        self,
        atlas,
        origin,
        throughput,
        batch_size,
        image_key=None,
        reduction=None,
        argmap=None,
        influx=None,
        efflux=None,
        lines=None,
        transmit_filters=None,
        receive_filters=None,
        skip_local=False,
        nonlocal_argmap=None,
    ):
        reduction = reduction or 'mean'
        image_key = image_key or 'images'
        argmap = argmap or (lambda x: ModelArgument(input=x))
        gradient = partial(atlas_gradient, atlas=atlas)
        accfn = partial(atlas_accfn, atlas=atlas, argmap=argmap)
        local_argmap = self.argmap
        influx = influx or (
            lambda arg: UnpackingModelArgument(input=arg.images))
        super().__init__(
            model=atlas,
            accfn=accfn,
            gradient=gradient,
            origin=origin,
            retain_dims=(-1, -2),
            throughput=throughput,
            batch_size=batch_size,
            reduction=reduction,
            params=None,
            influx=influx,
            efflux=efflux,
            lines=lines,
            transmit_filters=transmit_filters,
            receive_filters=receive_filters,
            skip_local=skip_local,
            local_argmap=local_argmap,
            nonlocal_argmap=nonlocal_argmap,
        )
        self.coors = {}
        self.masks = {}
        self.ref = atlas.atlas
        self.image_key = image_key
        for name in self.model.preweight.keys():
            compartment = self.ref.compartments[name]
            self.masks[name] = compartment[atlas.mask]
            self.coors[name] = self.ref.coors[self.masks[name]].t()

    def argmap(self, input, atlas):
        images = input[self.image_key]
        # Note that we require a second forward pass to get our arg.
        # Set skip_local if you don't need it.
        output = self.model(images)
        inputs = {
            name : apply_mask(images, mask, -2)
            for name, mask in self.masks.items()
        }
        inputs = ModelArgument(**inputs)
        weights = ModelArgument(**self.model.weight)
        preweights = ModelArgument(**self.model.preweight)
        coors = ModelArgument(**self.coors)
        return ModelArgument(
            input=input,
            ts=inputs,
            output=output,
            preweight=preweights,
            weight=weights,
            coor=coors
        )

    def forward(self):
        return super().forward(atlas=self.model)
