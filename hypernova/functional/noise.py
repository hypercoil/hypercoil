# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Noise
~~~~~
Noise sources.
"""
import math
import torch


def spsd_noise(dim, rank=None, std=0.05):
    rank = rank or dim[-1]
    noise = torch.empty((*dim, rank))
    var = std / math.sqrt(rank + (rank ** 2) / dim[-1])
    noise.normal_(std=math.sqrt(var))
    return noise @ noise.transpose(-1, -2)
