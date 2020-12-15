# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Domains
~~~~~~~
Functional image and preimage mappers.
"""
import torch


class _OutOfDomainHandler(object):
    def __init__(self):
        pass

    def test(self, x, bound):
        return torch.logical_and(
            x <= bound[-1],
            x >= bound[0]
        )


class Clip(_OutOfDomainHandler):
    def __init__(self):
        super(Clip, self).__init__()

    def apply(self, x, bound):
        out = x.detach().clone()
        out[out > bound[-1]] = bound[-1]
        out[out < bound[0]] = bound[0]
        return out


class Normalise(_OutOfDomainHandler):
    def __init__(self):
        super(Normalise, self).__init__()

    def apply(self, x, bound, axis=None):
        # This annoying conditional is necessary despite torch documentation
        # suggesting the contrary:
        # https://pytorch.org/docs/stable/tensors.html
        #
        # ctrl+f `mean` for the incorrect default signature that raises:
        # RuntimeError: Please look up dimensions by name, got: name = None.
        #
        # It could hardly be handled worse.
        out = x.detach().clone()
        if axis is None:
            upper = out.max()
            lower = out.min()
            unew = torch.minimum(bound[-1], out.max())
            lnew = torch.maximum(bound[0], out.min())
            out -= out.mean()
            out /= ((upper - lower) / (unew - lnew))
            out += (lnew - out.min())
        else:
            upper = out.max(axis)
            lower = out.min(axis)
            unew = torch.minimum(bound[-1], out.max(axis))
            lnew = torch.maximum(bound[0], out.min(axis))
            out -= out.mean(axis)
            out /= ((upper - lower) / (unew - lnew))
            out += (lnew - out.min(axis))
        return out
