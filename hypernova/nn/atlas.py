# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlas layer
~~~~~~~~~~~
Modules that linearly map voxelwise signals to labelwise signals.
"""
import torch
from torch.nn import Module, Parameter
from torch.distributions import Bernoulli
from ..functional.domain import Identity
from ..functional.noise import UnstructuredDropoutSource
from ..init.atlas import atlas_init_


class AtlasLinear(Module):
    """
    Time series extraction from an atlas via a linear map.

    Dimension
    ---------
    - Input: :math:`(N, *, X, Y, Z, T)` or :math:`(N, *, V, T)`
      N denotes batch size, `*` denotes any number of intervening dimensions,
      X, Y, and Z denote 3 spatial dimensions, V denotes total number of
      voxels, T denotes number of time points or observations.
    - Output: :math:`(N, *, L, T)`
      L denotes number of labels in the provided atlas.

    Parameters
    ----------
    atlas : Atlas object
        A neuroimaging atlas, implemented as a `DiscreteAtlas` or
        `ContinuousAtlas` object (`hypernova.init.DiscreteAtlas` and
        `hypernova.init.ContinuousAtlas`). This initialises the atlas labels
        from which representative time series are extracted.
    kernel_sigma : float
        If this is a float, then a Gaussian smoothing kernel with the
        specified width is applied to each label at initialisation.
    noise_sigma : float
        If this is a float, then Gaussian noise with the specified standard
        deviation is added to the label at initialisation.
    mask_input : bool
        Indicates that each input is a 4D image that must be masked before
        time series extraction. If True, then the boolean tensor stored in the
        `mask` field of the `atlas` input is used to subset and "unfold" each
        4D image into a 2D space by time matrix.
    spatial_dropout : float in [0, 1) (default 0)
        Probability of dropout for each voxel. If this is nonzero, then during
        training each voxel's weight has some probability of being set to zero,
        thus discounting the voxel from the time series estimate. In theory,
        this can promote learning a weight that is robust to any single voxel.
    min_voxels : positive int (default 1)
        Minimum number of voxels that each region must contain after dropout.
        If a random dropout results in fewer remaining voxels, then another
        random dropout will be sampled until the criterion is satisfied. Has no
        effect if `spatial_dropout` is zero.
    domain : Domain object (default Identity)
        A domain object from `hypernova.functional.domain`, used to specify
        the domain of the atlas weights. An `Identity` object yields the raw
        atlas weights, while an `Atanh` object constrains weights to (-a, a),
        and a `Logit` object constrains weights to (0, a) by transforming the
        raw weights through a tanh or sigmoid function, respectively. Using an
        appropriate domain can ensure that weights are nonnegative and that
        they do not grow explosively.
    reduce : 'mean', 'absmean', 'zscore', 'psc', or 'sum' (default 'mean')
        Strategy for reducing across voxels and generating a representative
        time series for each label.
        * `sum`: Sum over voxel time series.
        * `mean`: Compute the mean over voxel time series.
        * `absmean`: Compute the mean over voxel time series, treating any
          negative voxel weights as though they were positive.
        * `zscore`: Transform the sum of time series such that its temporal
          mean is 0 and its temporal standard deviation is 1.
        * `psc`: Transform the time series such that its value indicates the
          percent signal change from the mean.

    Attributes
    ----------
    preweight : Tensor :math:`(L, V)`
        Atlas map in the module's domain. L denotes the number of labels, and V
        denotes the number of voxels. Identical to `weight` if the domain is
        Identity.
    weight : Tensor :math:`(L, V)`
        Representation of the atlas as a linear map from voxels to labels,
        applied independently to each time point in each input image.
    postweight : Tensor :math:`(L, V)`
        Atlas map after application of spatial dropout. Spatial dropout has
        a chance of randomly removing each voxel from consideration when
        extracting each time series. Spatial dropout is applied only during
        training. Identical to `weight` if there is no spatial dropout.
    mask : Tensor :math:`(X, Y, Z)`
        Boolean-valued tensor indicating the voxels that should be included as
        inputs to the atlas transformation.
    """
    def __init__(self, atlas, kernel_sigma=None, noise_sigma=None,
                 mask_input=True, spatial_dropout=0, min_voxels=1,
                 domain=None, reduce='mean'):
        super(AtlasLinear, self).__init__()

        self.atlas = atlas
        self.kernel_sigma = kernel_sigma
        self.noise_sigma = noise_sigma
        self.domain = domain or Identity()
        self.reduction = reduce
        self.mask_input = mask_input
        self.mask = (torch.from_numpy(self.atlas.mask)
                     if self.mask_input else None)
        self.preweight = Parameter(torch.Tensor(
            self.atlas.n_labels, self.atlas.n_voxels
        ))
        self._configure_spatial_dropout(spatial_dropout, min_voxels)
        self.reset_parameters()

    def reset_parameters(self):
        atlas_init_(tensor=self.preweight,
                    atlas=self.atlas,
                    kernel_sigma=self.kernel_sigma,
                    noise_sigma=self.noise_sigma,
                    domain=self.domain,
                    normalise=False)

    def _configure_spatial_dropout(self, dropout_rate, min_voxels):
        if dropout_rate > 0:
            self.dropout = UnstructuredDropoutSource(
                distr=Bernoulli(torch.Tensor([1 - dropout_rate])),
                sample_axes=[-1]
            )
        else:
            self.dropout = None
        self.min_voxels = min_voxels

    @property
    def weight(self):
        return self.domain.image(self.preweight)

    @property
    def postweight(self):
        if self.dropout is not None:
            while True:
                n_voxels = (self.weight > 0).sum(-1)
                weight = self.dropout(self.weight)
                n_voxels = (weight > 0).sum(-1)
                if torch.all(n_voxels >= self.min_voxels):
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

    def apply_mask(self, input):
        shape = input.size()
        mask = self.mask
        extra_dims = 0
        while mask.dim() < input.dim() - 1:
            mask = mask.unsqueeze(0)
            extra_dims += 1
        input = input[mask.expand(shape[:-1])]
        input = input.view(*shape[:extra_dims], -1 , shape[-1])
        return input

    def forward(self, input):
        if self.mask_input:
            input = self.apply_mask(input)
        return self.reduce(input, self.postweight)

    def __repr__(self):
        s = f'{type(self).__name__}(atlas={self.atlas}, '
        s += f'mask={self.mask_input}, reduce={self.reduction})'
        return s
