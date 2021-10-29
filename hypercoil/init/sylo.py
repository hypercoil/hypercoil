# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Initialisations for sylo-based neural networks.
"""
import torch
import math
from torch.nn.init import calculate_gain


#TODO: This needs a lot of review/revision with a proper derivation. Right now
# I think we're attributing to symmetry lots of consequences of PSD.


def sylo_init_(tensors, a=0, mode='fan_in', init='uniform',
               nonlinearity='leaky_relu', symmetry=False):
    """Kaiming-like initialisation for sylo module.

    Notes
    -----
    * The overall fan is the square root of the product of
      (1) the fan from raw weights (H x r, W x r) to expanded template (H x W),
      (2) the fan from expanded template to the output feature map.
    * Fan (1) is the rank r of the raw weights. If the operation is symmetric
      (the left and right weights are identical), we need to add an additional
      term that scales quadratically with the rank and inversely with the raw
      weight dimension (H = W). It's possible that these are the first two
      terms of a longer series, but I'm not currently sure how to figure this
      out.
    * Fan (2) is computed as for convolutional networks: it's the product of
      the number of channels and the receptive field size, which here is the
      size of the crosshair (H + W - 1).
    * I hacked this out empirically (don't do this...), so there's a good
      chance that we can do better once we come up with a good theoretical
      justification.
    * I believe that we need to take a double square root because the template
      is a product of the raw weights.
    """
    gain = calculate_gain(nonlinearity, a)
    fan_crosshair = _calculate_correct_fan_crosshair(tensors, mode)
    fan_expansion = _calculate_fan_in_expansion(tensors, symmetry)
    # TODO: Does gain go inside or outside of the outer sqrt?
    # Right now it's outside since we'd rather the std explode than vanish...
    std = gain / math.sqrt(math.sqrt(fan_crosshair * fan_expansion))
    if init == 'normal':
        for tensor in tensors:
            with torch.no_grad():
                tensor.normal_(0, std)
        return

    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    for tensor in tensors:
        with torch.no_grad():
            tensor.uniform_(-bound, bound)


def _calculate_correct_fan_crosshair(tensors, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError('Mode {} not supported, please use one of '
                         '{}'.format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out_crosshair(tensors)
    return fan_in if mode == 'fan_in' else fan_out


def _calculate_fan_in_and_fan_out_crosshair(tensors):
    num_input_fmaps = tensors[0].size(1)
    num_output_fmaps = tensors[0].size(0)
    receptive_field_size = tensors[0].size(-2) + tensors[1].size(-2) - 1
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_fan_in_expansion(tensors, symmetry=False):
    L, R = tensors
    rank = tensors[0].size(-1)
    if symmetry is True:
        dim = tensors[0].size(-2)
        return rank + (rank ** 2) / dim
    else:
        return rank


def sparse_kaiming_uniform_(tensor, mask=None, a=0, mode='fan_in',
                            nonlinearity='leaky_relu'):
    """Kaiming uniform initialisation for a linear weight that has many terms
    fixed at zero.
    """
    if mask is None:
        mask = tensor
    fan = {}
    pathways = (mask != 0)
    fan['fan_in'] = pathways.sum(1).float().mean()
    fan['fan_out'] = pathways.sum(0).float().mean()
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan[mode])
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        tensor.uniform_(-bound, bound)
