# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Null-option multi-logit
~~~~~~~~~~~~~~~~~~~~~~~
Softmax domain mapper with a null option. Needs its own submodule so we don't
get a circular import.
"""
import torch
from .domainbase import _Domain, _PhaseAmplitudeDomain
from ..init.base import ConstantInitialiser


class Atanh(_Domain):
    """
    Hyperbolic tangent domain mapper. Constrain tensor values between some
    finite scale value and its negation.

    Parameters/Attributes
    ---------------------
    scale : float (default 1)
        Maximum/minimum value attained by the hyperbolic tangent map. Scale
        factor applied to the image. The multiplicative inverse (reciprocal) is
        applied before the preimage is computed.
    limits : (float min, float max) (default -3, 3)
        Minimum and maximum values in the preimage itself. Used for two
        purposes: avoiding infinities when the tensor's values include the
        supremum or infimum of an asymptotic function (i.e., -`scale` or
        `scale`) and restricting parameter values to a range where the gradient
        has not vanished.
    handler : _OutOfDomainHandler object (default Clip)
        Object specifying a method for handling out-of-domain entries.
    """
    def __init__(self, scale=1, handler=None, limits=(-3, 3)):
        super(Atanh, self).__init__(
            handler=handler, bound=(-scale, scale),
            scale=scale, limits=limits)
        self.preimage_map = torch.atanh
        self.image_map = torch.tanh


class AmplitudeAtanh(_PhaseAmplitudeDomain, Atanh):
    """
    Hyperbolic tangent amplitude domain mapper. Constrain the amplitudes of a
    complex-valued tensor between zero and some finite scale value.

    Parameters/Attributes
    ---------------------
    scale : float (default 1)
        Maximum/minimum value attained by the hyperbolic tangent map. Scale
        factor applied to the image. The multiplicative inverse (reciprocal) is
        applied before the preimage is computed. (Note that the amplitude used
        as the argument to the hyperbolic tangent is never negative and so the
        effective minimum is zero.)
    limits : (float min, float max) (default -3, 3)
        Minimum and maximum values in the preimage itself. Used for two
        purposes: avoiding infinities when the tensor's values include the
        supremum or infimum of an asymptotic function (i.e., `scale`) and
        restricting parameter values to a range where the gradient has not
        vanished.
    handler : _OutOfDomainHandler object (default Clip)
        Object specifying a method for handling out-of-domain entries.
    """
    pass


class Logit(_Domain):
    """
    Logistic/sigmoid domain mapper. Constrain tensor values between 0 and some
    finite scale value.

    Parameters/Attributes
    ---------------------
    scale : float (default 1)
        Maximum value attainable by the logistic map. Scale factor applied to
        the image. The multiplicative inverse (reciprocal) is applied before
        the preimage is computed.
    limits : (float min, float max) (default -4.5, 4.5)
        Minimum and maximum values in the preimage itself. Used for two
        purposes: avoiding infinities when the tensor's values include the
        supremum or infimum of an asymptotic function (i.e., 0 or `scale`) and
        restricting parameter values to a range where the gradient has not
        vanished.
    handler : _OutOfDomainHandler object (default Clip)
        Object specifying a method for handling out-of-domain entries.
    """
    def __init__(self, scale=1, loc=None, handler=None, limits=(-4.5, 4.5)):
        loc = loc or (scale / 2)
        shift = loc - scale / 2
        super(Logit, self).__init__(
            handler=handler, bound=(loc - scale / 2, loc + scale / 2),
            loc=shift, scale=scale, limits=limits)
        self.preimage_map = torch.logit
        self.image_map = torch.sigmoid


class MultiLogit(_Domain):
    """
    Softmax domain mapper. Maps between a multinomial logit space and
    estimated class probabilities in the probability simplex.

    The forward function is a softmax. Note that the softmax function does
    not have a unique inverse; here we use the elementwise natural logarithm
    as an 'inverse'.

    Parameters/Attributes
    ---------------------
    axis : int (default -1)
        Axis of tensors in the domain along which 1D slices are mapped to the
        probability simplex.
    minim : nonnegative float (default 1e-3)
        Before it is mapped to its preimage, an input is bounded to the closed
        interval [`minim`, 1 - `minim`]. This serves two purposes: avoiding
        infinities when the tensor's values include the supremum or infimum
        (0 and 1) and restricting parameter values to a range where the
        gradient has not vanished.
    handler : _OutOfDomainHandler object (default Clip)
        Object specifying a method for handling out-of-domain entries.
    """
    def __init__(self, axis=-1, minim=1e-3, handler=None):
        super(MultiLogit, self).__init__(
            handler=handler, bound=(minim, 1 - minim))
        self.axis = axis
        self.preimage_map = torch.log
        self.image_map = lambda x: torch.softmax(x, self.axis)


class AmplitudeMultiLogit(_PhaseAmplitudeDomain, MultiLogit):
    """
    Softmax amplitude domain mapper. Maps the amplitudes of a complex-valued
    tensor between a multinomial logit space and estimated class probabilities
    in the probability simplex.

    The forward function is a softmax applied to the amplitudes. Note that the
    softmax function does not have a unique inverse; here we use the
    elementwise natural logarithm as an 'inverse'.

    Parameters/Attributes
    ---------------------
    axis : int (default -1)
        Axis of tensors in the domain along which 1D slices are mapped to the
        probability simplex.
    minim : nonnegative float (default 1e-3)
        Before it is mapped to its preimage, an input is bounded to the closed
        interval [`minim`, 1 - `minim`]. This serves two purposes: avoiding
        infinities when the tensor's values include the supremum or infimum
        (0 and 1) and restricting parameter values to a range where the
        gradient has not vanished.
    handler : _OutOfDomainHandler object (default Clip)
        Object specifying a method for handling out-of-domain entries.
    """
    pass


class NullOptionMultiLogit(_Domain):
    def __init__(self, axis=-1, minim=1e-3, null_init=None, handler=None):
        super(NullOptionMultiLogit, self).__init__(
            handler=handler, bound=(minim, 1 - minim))
        self.axis = axis
        self.null_init = null_init or ConstantInitialiser(0)
        self.signature[axis] = (
            lambda x: x + 1,
            lambda x: x - 1
        )

        def preimage_map(x):
            dim = list(x.size())
            dim[self.axis] = 1
            nulls = torch.empty(dim)
            self.null_init(nulls)
            nulls = self.handler.apply(nulls, self.bound)
            z = torch.cat((x, nulls), self.axis)
            return torch.log(z)

        def image_map(x):
            x = torch.softmax(x, self.axis)
            return x.index_select(
                self.axis,
                torch.arange(0, x.size(self.axis) - 1)
            )

        self.preimage_map = preimage_map
        self.image_map = image_map


class ANOML(_PhaseAmplitudeDomain, NullOptionMultiLogit):
    pass
