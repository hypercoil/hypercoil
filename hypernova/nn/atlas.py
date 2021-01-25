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
    def __init__(self, atlas, kernel_sigma=None, noise_sigma=None):
        super(AtlasLinear, self).__init__()

        self.atlas = atlas
        self.kernel_sigma = kernel_sigma
        self.noise_sigma = noise_sigma
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
        return self.weight @ input
