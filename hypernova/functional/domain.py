# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Domains
~~~~~~~
Functional image and preimage mappers.
"""
import torch
from .activation import complex_decompose, complex_recompose


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
        bound = torch.Tensor(bound)
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
        bound = torch.Tensor(bound)
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


class _Domain(object):
    def __init__(self, handler=None, bound=None, scale=1, limits=None):
        self.handler = handler or Clip()
        bound = bound or [-float('inf'), float('inf')]
        limits = limits or [-float('inf'), float('inf')]
        self.bound = torch.Tensor(bound)
        self.limits = torch.Tensor(limits)
        self.scale = scale

    def test(self, x):
        return self.handler.test(x, self.bound)

    def handle_ood(self, x):
        return self.handler.apply(x)

    def preimage(self, x):
        x = self.handler.apply(x, self.bound)
        i = self.preimage_map(x / self.scale)
        i = self.handler.apply(i, self.limits)
        return i

    def image(self, x):
        return self.scale * self.image_map(x)


class _PhaseAmplitudeDomain(_Domain):
    def preimage(self, x):
        ampl, phase = complex_decompose(x)
        ampl = super(_PhaseAmplitudeDomain, self).preimage(ampl)
        return complex_recompose(ampl, phase)

    def image(self, x):
        ampl, phase = complex_decompose(x)
        ampl = super(_PhaseAmplitudeDomain, self).image(ampl)
        return complex_recompose(ampl, phase)


class Identity(_Domain):
    def preimage(self, x):
        return x

    def image(self, x):
        return x


class Linear(_Domain):
    def __init__(self, scale=1):
        super(Linear, self).__init__()
        self.scale = scale

    def preimage(self, x):
        return x / self.scale

    def image(self, x):
        return self.scale * x


class Logit(_Domain):
    def __init__(self, scale=1, handler=None, limits=(-4.5, 4.5)):
        super(Logit, self).__init__(
            handler=handler, bound=(0, scale),
            scale=scale, limits=limits)
        self.preimage_map = torch.logit
        self.image_map = torch.sigmoid


class Atanh(_Domain):
    def __init__(self, scale=1, handler=None, limits=(-3, 3)):
        super(Atanh, self).__init__(
            handler=handler, bound=(-scale, scale),
            scale=scale, limits=limits)
        self.preimage_map = torch.atanh
        self.image_map = torch.tanh


class AmplitudeAtanh(_PhaseAmplitudeDomain, Atanh):
    pass
