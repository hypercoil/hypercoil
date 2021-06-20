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
    """
    System for evaluating and modifying out-of-domain entries prior to preimage
    mapping to ensure that `nan` values are not introduced.

    Methods
    -------
    test(x, bound)
        Evaluate whether each entry in a tensor falls within bounds.

        Parameters
        ----------
        x : Tensor
            Tensor whose entries should be evaluated.
        bound : (float min, float max)
            Minimum and maximum permitted values for the test to return True.

        Returns
        -------
        result : Tensor
            Boolean-valued tensor indicating whether each entry of the input is
            in the prescribed bounds.
    """
    def __init__(self):
        pass

    def __repr__(self):
        return f'{type(self).__name__}()'

    def test(self, x, bound):
        return torch.logical_and(
            x <= bound[-1],
            x >= bound[0]
        )


class Clip(_OutOfDomainHandler):
    """
    Handle out-of-domain values by clipping them to the closest allowed point.

    Methods
    -------
    apply(x, bound)
        Clip values in the specified tensor to the specified bounds.

        Parameters
        ----------
        x : Tensor
            Tensor whose out-of-domain entries are to be clipped.
        bound : (float min, float max)
            Minimum and maximum permitted values; values greater than the
            maximum will be clipped to the maximum and values less than the
            minimum will be clipped to the minimum.

        Returns
        -------
        out : Tensor
            Copy of the input tensor with out-of-domain entries clipped.

    test(x, bound)
        Evaluate whether each entry in a tensor falls within bounds.

        Parameters
        ----------
        x : Tensor
            Tensor whose entries should be evaluated.
        bound : (float min, float max)
            Minimum and maximum permitted values. Any values outside the
            prescribed interval are clipped.

        Returns
        -------
        result : Tensor
            Boolean-valued tensor indicating whether each entry of the input is
            in the prescribed bounds.

    ood : 'clip' or 'norm' (default `clip`)
        Indicates how the initialisation handles out-of-domain values.
        `clip` indicates that out-of-domain values should be clipped to the
        closest allowed point and `norm` indicates that the entire spectrum
        (in-domain and out-of-domain values) should be re-scaled so that it
        fits in the domain bounds (not recommended).
    """
    def apply(self, x, bound):
        out = x.detach().clone()
        bound = torch.Tensor(bound)
        out[out > bound[-1]] = bound[-1]
        out[out < bound[0]] = bound[0]
        return out


class Normalise(_OutOfDomainHandler):
    """
    Normalise entries in the specified tensor so that all are in an interval
    specified by bounds.

    Note that this handler is in general much more radical than clipping. In
    particular, if any entry at all is out of bounds, nearly all entries in
    the tensor will be edited, and a single extreme outlier can destroy the
    variance in the dataset. If you are considering using this because most of
    the data you expect to see will be outside of the prescribed domain,
    consider using a different domain first (for instance, using the `scale`
    parameter to accommodate a larger feasible interval).

    The normalisation procedure works by mapping the original range of
    observations, [obs_min, obs_max], to
    [max(obs_min, lbound), min(obs_max, ubound)] while preserving relative
    distances between observations.

    Methods
    -------
    apply(x, bound)
        Re-scale all values in the specified tensor to fall in the specified
        interval.

        Parameters
        ----------
        x : Tensor
            Tensor whose out-of-domain entries are to be normalised.
        bound : (float min, float max)
            Minimum and maximum permitted values.

        Returns
        -------
        out : Tensor
            Copy of the input tensor with all entries normalised.

    test(x, bound)
        Evaluate whether each entry in a tensor falls within bounds.

        Parameters
        ----------
        x : Tensor
            Tensor whose entries should be evaluated.
        bound : (float min, float max)
            Minimum and maximum permitted values. Any values outside the
            prescribed interval are clipped.

        Returns
        -------
        result : Tensor
            Boolean-valued tensor indicating whether each entry of the input is
            in the prescribed bounds.
    """
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


class _Domain(torch.nn.Module):
    """
    Functional domain mapper.

    Map a tensor to either its image or its preimage under some function in a
    safe manner that handles out-of-domain entries to avoid nan and error
    values. Used as a superclass for parameter-constraining mappers.

    Parameters/Attributes
    ---------------------
    scale : float (default 1)
        Scale factor applied to the image under the specified function. Can be
        used to relax or tighten bounds.
    bound : (float min, float max) (default -inf, inf)
        Minimum and maximum tolerated values in the input to the preimage
        mapper. Subclasses do not expose this parameter.
    limits : (float min, float max) (default -inf, inf)
        Minimum and maximum values in the preimage itself. Used for two
        purposes: avoiding infinities when the tensor's values include the
        supremum or infimum of an asymptotic function and restricting parameter
        values to a range where the gradient has not vanished.
    handler : _OutOfDomainHandler object (default Clip)
        Object specifying a method for handling out-of-domain entries.

    Methods
    -------
    preimage(x)
        Map a tensor to its preimage under the transformation. Any values
        outside the transformation's range are first handled.

    image(x)
        Map a tensor to its image under the transformation.

    handle_ood(x)
        Apply an out-of-domain handler to ensure that all tensor entries are
        within bounds.

    test(x)
        Evaluate whether each entry in a tensor falls within bounds.
    """
    def __init__(self, handler=None, bound=None, loc=0, scale=1, limits=None):
        super(_Domain, self).__init__()
        self.handler = handler or Clip()
        bound = bound or [-float('inf'), float('inf')]
        limits = limits or [-float('inf'), float('inf')]
        self.bound = torch.Tensor(bound)
        self.limits = torch.Tensor(limits)
        self.loc = loc
        self.scale = scale
        self.signature = {}

    def extra_repr(self):
        s = []
        if self.scale != 1:
            s += [f'scale={self.scale}']
        if not torch.all(torch.isinf(self.bound)):
            s += [f'bound=({self.bound[0]}, {self.bound[1]})']
            s += [f'handler={self.handler.__repr__()}']
        return ', '.join(s)

    def test(self, x):
        return self.handler.test(x, self.bound)

    def handle_ood(self, x):
        return self.handler.apply(x)

    def preimage(self, x):
        x = self.handler.apply(x, self.bound)
        i = self.preimage_map((x - self.loc) / self.scale)
        i = self.handler.apply(i, self.limits)
        return i

    def image(self, x):
        return self.scale * self.image_map(x) + self.loc

    def preimage_dim(self, dim):
        if isinstance(dim, torch.Tensor):
            dim = list(dim.shape)
        for k, (v, _) in self.signature.items():
            dim[k] = v(dim[k])
        return torch.Size(dim)

    def image_dim(self, dim):
        if isinstance(dim, torch.Tensor):
            dim = list(dim.shape)
        for k, (_, v) in self.signature.items():
            dim[k] = v(dim[k])
        return torch.Size(dim)


class _PhaseAmplitudeDomain(_Domain):
    """
    Phase/amplitude functional domain mapper.

    Map the amplitude of a complex-valued tensor to either its image or its
    preimage under some function in a safe manner that handles out-of-domain
    entries to avoid nan and error values. Used as a superclass for parameter-
    constraining mappers.

    Parameters/Attributes
    ---------------------
    scale : float (default 1)
        Scale factor applied to the image under the specified function. Can be
        used to relax or tighten bounds.
    bound : (float min, float max) (default -inf, inf)
        Minimum and maximum tolerated values in the input to the preimage
        mapper. Subclasses do not expose this parameter.
    limits : (float min, float max) (default -inf, inf)
        Minimum and maximum values in the preimage itself. Used for two
        purposes: avoiding infinities when the tensor's values include the
        supremum or infimum of an asymptotic function and restricting parameter
        values to a range where the gradient has not vanished.
    handler : _OutOfDomainHandler object (default Clip)
        Object specifying a method for handling out-of-domain entries.

    Methods
    -------
    preimage(x)
        Map the amplitude of a complex-valued tensor to its preimage under the
        transformation. Any values outside the transformation's range are first
        handled.

    image(x)
        Map the amplitude of a complex-valued a tensor to its image under the
        transformation.

    handle_ood(x)
        Apply an out-of-domain handler to ensure that all tensor entries are
        within bounds.

    test(x)
        Evaluate whether each entry in a tensor falls within bounds.
    """
    def preimage(self, x):
        ampl, phase = complex_decompose(x)
        ampl = super(_PhaseAmplitudeDomain, self).preimage(ampl)
        return complex_recompose(ampl, phase)

    def image(self, x):
        ampl, phase = complex_decompose(x)
        ampl = super(_PhaseAmplitudeDomain, self).image(ampl)
        return complex_recompose(ampl, phase)


class Identity(_Domain):
    """
    Identity domain mapper.

    Methods
    -------
    preimage(x)
        Returns exactly the unmodified input.

    image(x)
        Returns exactly the unmodified input.

    handle_ood(x)
        Does nothing.

    test(x)
        Returns True for all finite numerical entries.
    """
    def preimage(self, x):
        return x

    def image(self, x):
        return x


class Linear(_Domain):
    """
    Linear domain mapper.

    Parameters/Attributes
    ---------------------
    scale : float (default 1)
        Slope of the linear map. Scale factor applied to the image. The
        multiplicative inverse (reciprocal) is applied before the preimage is
        computed.

    Methods
    -------
    preimage(x)
        Map a tensor to its preimage under the linear transformation.

    image(x)
        Map a tensor to its image under the linear transformation.

    handle_ood(x)
        Does nothing.

    test(x)
        Returns True for all finite numerical entries.
    """
    def __init__(self, scale=1):
        super(Linear, self).__init__()
        self.scale = scale

    def preimage(self, x):
        return x / self.scale

    def image(self, x):
        return self.scale * x


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

    Methods
    -------
    preimage(x)
        Map a tensor to its preimage under the sigmoid transformation (using
        the logit function).

    image(x)
        Map a tensor to its image under the sigmoid transformation.

    handle_ood(x)
        Handles tensor values that are outside of the scaled sigmoid's range
        (0, `scale`).

    test(x)
        Indicates whether each entry of a tensor is in the range of the scaled
        sigmoid (0, `scale`).
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

    Methods
    -------
    preimage(x)
        Map a tensor to its preimage under the hyperbolic tangent
        transformation (using the atanh function).

    image(x)
        Map a tensor to its image under the hyperbolic tangent transformation.

    handle_ood(x)
        Handles tensor values that are outside of the scaled hyperbolic tangent
        range (-`scale`, `scale`).

    test(x)
        Indicates whether each entry of a tensor is in the range of the scaled
        hyperbolic tangent (-`scale`, `scale`).
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

    Methods
    -------
    preimage(x)
        Map the amplitude of a complex-valued tensor to its preimage under the
        hyperbolic tangent transformation (using the atanh function).

    image(x)
        Map the amplitude of a complex-valued tensor to its image under the
        hyperbolic tangent transformation.

    handle_ood(x)
        Handles tensor values that are outside of the scaled hyperbolic tangent
        range (-`scale`, `scale`).

    test(x)
        Indicates whether each entry of a tensor is in the range of the scaled
        hyperbolic tangent (-`scale`, `scale`).
    """
    pass
