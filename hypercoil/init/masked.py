# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Initialisations involving boolean masks.
"""
import torch
import math
from torch.nn.init import calculate_gain


# We need to go back and check these imported inits at some point -- they
# are relics from an older era.
def sparse_kaiming_uniform_(tensor, mask=None, a=0, mode='fan_in',
                            nonlinearity='leaky_relu'):
    """
    Kaiming uniform initialisation for a linear weight that has many terms
    fixed at zero according to a boolean mask.
    """
    if mask is None:
        mask = tensor
    pathways = (mask != 0)
    fan = {
        'fan_in' : pathways.sum(1).float().mean(),
        'fan_out' : pathways.sum(0).float().mean()
    }
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan[mode])
    # Calculate uniform bounds from standard deviation
    bound = math.sqrt(3.0) * std
    with torch.no_grad():
        tensor.uniform_(-bound, bound)
