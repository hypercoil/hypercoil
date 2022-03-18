# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Domains
~~~~~~~
Functional image and preimage mappers.
"""
import torch
from .domainbase import (
    Clip, Normalise,
    _Domain, _PhaseAmplitudeDomain,
    Identity, Linear, Affine
)


class Atanh(_Domain):
    """
    Hyperbolic tangent domain mapper. Constrain tensor values between some
    finite scale value and its negation.

    Parameters
    ----------
    scale : float (default 1)
        Maximum/minimum value attained by the hyperbolic tangent map. Scale
        factor applied to the image. The multiplicative inverse (reciprocal)
        is applied before the preimage is computed.
    limits : (float min, float max) (default -3, 3)
        Minimum and maximum values in the preimage itself. Used for two
        purposes: avoiding infinities when the tensor's values include the
        supremum or infimum of an asymptotic function (i.e., -`scale` or
        `scale`) and restricting parameter values to a range where the
        gradient has not vanished.
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

    Parameters
    ----------
    scale : float (default 1)
        Maximum/minimum value attained by the hyperbolic tangent map. Scale
        factor applied to the image. The multiplicative inverse (reciprocal)
        is applied before the preimage is computed. (Note that the amplitude
        used as the argument to the hyperbolic tangent is never negative and
        so the effective minimum is zero.)
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

    Parameters
    ----------
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
        if loc is None: loc = (scale / 2)
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
    as an 'inverse'. For a relatively well-behaved map, use together with
    `init.dirichlet.DirichletInit`

    Parameters
    ----------
    axis : int (default -1)
        Axis of tensors in the domain along which 1D slices are mapped to the
        probability simplex.
    minim : nonnegative float (default 1e-3)
        Before it is mapped to its preimage, an input is bounded to the closed
        interval [`minim`, 1 - `minim`]. This serves two purposes: avoiding
        infinities when the tensor's values include the supremum or infimum
        (0 and 1) and restricting parameter values to a range where the
        gradient has not vanished.
    smoothing : nonnegative float (default 0)
        For use in preimage mapping. Increasing the smoothing will result in
        a higher-entropy / more equiprobable image.
    handler : _OutOfDomainHandler object (default Clip)
        Object specifying a method for handling out-of-domain entries.
    """
    def __init__(self, axis=-1, minim=1e-3, smoothing=0, handler=None):
        super(MultiLogit, self).__init__(
            handler=handler, bound=(minim, 1 - minim))
        self.axis = axis
        self.preimage_map = lambda x: torch.log(x + smoothing)
        self.image_map = lambda x: torch.softmax(x, self.axis)


class AmplitudeMultiLogit(_PhaseAmplitudeDomain, MultiLogit):
    """
    Softmax amplitude domain mapper. Maps the amplitudes of a complex-valued
    tensor between a multinomial logit space and estimated class probabilities
    in the probability simplex.

    The forward function is a softmax applied to the amplitudes. Note that the
    softmax function does not have a unique inverse; here we use the
    elementwise natural logarithm as an 'inverse'.

    Parameters
    ----------
    axis : int (default -1)
        Axis of tensors in the domain along which 1D slices are mapped to the
        probability simplex.
    minim : nonnegative float (default 1e-3)
        Before it is mapped to its preimage, an input's amplitude is bounded
        to the closed interval [`minim`, 1 - `minim`]. This serves two
        purposes: avoiding infinities when the tensor's amplitudes include the
        supremum or infimum (0 and 1) and restricting parameter values to a
        range where the gradient has not vanished.
    handler : _OutOfDomainHandler object (default Clip)
        Object specifying a method for handling out-of-domain entries.
    """
    pass


class NullOptionMultiLogit(_Domain):
    """
    Softmax domain mapper wherein the preimage contains an additional class
    corresponding to a null assignment. Any probability mass assigned to this
    class is swallowed in the forward transformation.

    The forward function is a softmax followed by deletion of the null option.
    Note that this function does not even come close to having a unique
    inverse; the `preimage_map` should really only be used for initialising
    weights in this domain.

    Parameters
    ----------
    axis : int (default -1)
        Axis of tensors in the domain along which 1D slices are mapped to the
        probability simplex.
    minim : nonnegative float (default 1e-4)
        Before it is mapped to its preimage, a (normalised) input is bounded
        to the closed interval [`minim`, 1 - `minim`]. This serves two
        purposes: avoiding infinities when the tensor's values include the
        supremum or infimum (0 and 1) and restricting parameter values to a
        range where the gradient has not vanished.
    buffer : nonnegative float (default 1e-4)
        Buffer added when computing the normalisation. Serves a similar
        purpose to `minim`. TODO: add a formula here...
    handler : _OutOfDomainHandler object (default Clip)
        Object specifying a method for handling out-of-domain entries.
    """
    def __init__(self, axis=-1, minim=1e-4, buffer=1e-4, handler=None):
        raise NotImplementedError
        super(NullOptionMultiLogit, self).__init__(
            handler=handler, bound=(minim, 1 - minim))
        self.axis = axis
        self.signature[axis] = (
            lambda x: x + 1,
            lambda x: x - 1
        )
        self.buffer = buffer

        def preimage_map(x):
            dim = list(x.size())
            dim[self.axis] = 1
            renorm = x.sum(self.axis, keepdim=True)
            maximum = torch.maximum(
                renorm.max() + self.buffer,
                torch.tensor(1.0, dtype=x.dtype, device=x.device)
            )
            nulls = maximum - renorm
            z = torch.cat((x, nulls), self.axis)
            z /= maximum
            z = self.handler.apply(z, self.bound)
            return torch.log(z)

        def image_map(x):
            x = torch.softmax(x, self.axis)
            return x.index_select(
                self.axis,
                torch.arange(
                    0, x.size(self.axis) - 1
                    dtype=x.dtype, device=x.device
                )
            )

        self.preimage_map = preimage_map
        self.image_map = image_map


class ANOML(_PhaseAmplitudeDomain, NullOptionMultiLogit):
    """
    Softmax amplitude domain mapper wherein the preimage also contains an
    additional class corresponding to a null assignment. Any probability mass
    assigned to this class is swallowed in the forward transformation.

    The forward function is a softmax applied to the amplitudes followed by
    deletion of the null option. Note that this function does not even come
    close to having a unique inverse; the `preimage_map` should really only
    be used for initialising weights in this domain.

    Parameters
    ----------
    axis : int (default -1)
        Axis of tensors in the domain along which 1D slices are mapped to the
        probability simplex.
    minim : nonnegative float (default 1e-3)
        Before it is mapped to its preimage, a (normalised) input's amplitude
        is bounded to the closed interval [`minim`, 1 - `minim`]. This serves
        two purposes: avoiding infinities when the tensor's amplitudes include
        the supremum or infimum (0 and 1) and restricting parameter values to
        a range where the gradient has not vanished.
    handler : _OutOfDomainHandler object (default Clip)
        Object specifying a method for handling out-of-domain entries.
    """
    def __init__(self, axis=-1, minim=1e-4, buffer=1e-4, handler=None):
        raise NotImplementedError
        super(ANOML, self).__init__(
            axis=axis,
            minim=minim,
            buffer=buffer,
            handler=handler
        )


def bandpass_iir(X, feature_ax=False):
    dim = X.shape[-1]
    if feature_ax:
        dim = X.shape[-2]
    mask = torch.ones(dim, dtype=X.dtype, device=X.device)
    mask[slice(1, None, 2)] = 0
    mask[slice(2, None, 4)] = -1
    if feature_ax:
        X = X + torch.flip(X, (-2,))
        mask = mask.unsqueeze(-1)
    else:
        X = X + torch.flip(X, (-1,))
    return mask * X


class IIRNumerator(_Domain):
    #TODO: fix limits
    # default limits based roughly on Kuznetsov init
    def __init__(self, handler=None, limits=(0, 1), peak=0.5,
                 btype='bandpass', order=1, feature_ax=False):
        super(IIRNumerator, self).__init__(
            handler=handler, bound=limits,
            loc=0, scale=1, limits=limits)
        self.btype = btype
        self.peak = peak
        self.order = order
        self.feature_ax = feature_ax
        self._init_mask()
        self.preimage_map = lambda x: self.encode(x)
        self.image_map = lambda x: self.iircoefs(x)

    def _init_mask(self):
        if self.btype == 'bandpass':
            self.mask = torch.ones(self.order * 2 + 1)
            self.mask[slice(1, None, 2)] = 0
            self.mask[slice(2, None, 4)] = -1
        elif self.btype == 'bandstop':
            self.mask = torch.ones(self.order * 2 + 1)
            if self.peak < 0.5:
                self.mask[slice(1, None, 2)] = -1
            elif self.peak == 0.5:
                self.mask[slice(1, None, 2)] = 0
        elif self.btype =='lowpass':
            self.mask = torch.ones(self.order + 1)
        elif self.btype =='highpass':
            self.mask = torch.ones(self.order + 1)
            self.mask[slice(1, None, 2)] = -1
        if self.feature_ax:
            self.mask = self.mask.unsqueeze(-1)

    def iircoefs(self, X):
        if self.feature_ax:
            X = X + torch.flip(X, (-2,))
        else:
            X = X + torch.flip(X, (-1,))
        return self.mask.type(X.dtype).to(X.device) * X / 2

    def encode(self, X):
        if self.btype == 'bandpass':
            y = torch.zeros(self.order * 2 + 1)
            y[slice(0, len(X) * 2, 2)] = X
        else:
            if self.btype == 'bandstop':
                y = torch.zeros(
                    self.order * 2 + 1,
                    dtype=X.dtype,
                    device=X.device
                )
            elif self.btype == 'lowpass' or self.btype == 'highpass':
                y = torch.empty(
                    self.order + 1,
                    dtype=X.dtype,
                    device=X.device
                )
            y[slice(0, len(X), 1)] = X
        y = y + torch.flip(y, (-1,))
        y[len(y) // 2] /= 2
        return torch.abs(y)


class IIRDenominator(_Domain):
    #TODO: fix limits
    # default limits based roughly on Kuznetsov init
    def __init__(self, handler=None, limits=(0, 0.5), peak=0.5,
                 btype='bandpass', order=1, feature_ax=False):
        super(IIRDenominator, self).__init__(
            handler=handler, bound=limits,
            loc=0, scale=1, limits=limits)
        self.btype = btype
        self.peak = peak
        self.order = order
        self.feature_ax = feature_ax
        self._init_mask()
        self.preimage_map = lambda x: self.encode(x)
        self.image_map = lambda x: self.iircoefs(x)

    def _init_mask(self):
        if self.btype == 'bandpass' or self.btype == 'bandstop':
            self.mask = torch.ones(self.order * 2)
        elif self.btype == 'highpass' or self.btype == 'highpass':
            self.mask = torch.ones(self.order)
        if self.peak < 0.5:
            self.mask[slice(0, None, 2)] = -1
        elif self.peak == 0.5:
            self.mask[slice(0, None, 2)] = 0
        if self.feature_ax:
            self.mask = self.mask.unsqueeze(-1)

    def iircoefs(self, X):
        return self.mask.type(X.dtype).to(X.device) * X

    def encode(self, X):
        return torch.abs(X)
