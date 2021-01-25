# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Atlas layer
~~~~~~~~~~~
Modules supporting filtering/convolution as a product in the frequency domain.
"""
import torch
from torch.nn import Module, Parameter
from ..init.atlas import atlas_init_


class AtlasLinear(Module):
    def __init__(self, atlas, kernel_sigma=None,
                 noise_sigma=None, mask_input=True):
        super(AtlasLinear, self).__init__()

        self.atlas = atlas
        self.kernel_sigma = kernel_sigma
        self.noise_sigma = noise_sigma
        self.mask_input = mask_input
        self.mask = (torch.from_numpy(self.atlas.mask)
                     if self.mask_input else None)
        self.weight = Parameter(torch.Tensor(
            self.atlas.n_labels, self.atlas.n_voxels
        ))
        self.reset_parameters()

    def reset_parameters(self):
        atlas_init_(tensor=self.weight,
                    atlas=self.atlas,
                    kernel_sigma=self.kernel_sigma,
                    noise_sigma=self.noise_sigma)

    def forward(self, input):
        if self.mask_input:
            shape = input.size()
            mask = self.mask
            extra_dims = 0
            while mask.dim() < input.dim() - 1:
                mask = mask.unsqueeze(0)
                extra_dims += 1
            input = input[mask.expand(shape[:-1])]
            input = input.view(*shape[:extra_dims], -1 , shape[-1])
        return self.weight @ input
