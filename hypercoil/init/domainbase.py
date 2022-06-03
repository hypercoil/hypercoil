# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Basic domains
~~~~~~~~~~~~~
Functional image and preimage mappers and supporting utilities.
"""
import math
import torch
from ..functional.utils import complex_decompose, complex_recompose


class _OutOfDomainHandler(object):
    """
    System for evaluating and modifying out-of-domain entries prior to preimage
    mapping to ensure that `nan` values are not introduced.
    """
    def __repr__(self):
        return f'{type(self).__name__}()'

    def test(self, x, bound):
        """
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
        return torch.logical_and(
            x <= bound[-1],
            x >= bound[0]
        )


class Clip(_OutOfDomainHandler):
    """
    Handle out-of-domain values by clipping them to the closest allowed point.
    """
    def apply(self, x, bound):
        """
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
        """
        out = x.detach().clone()
        bound = torch.tensor(bound, dtype=x.dtype, device=x.device)
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
    """
    def apply(self, x, bound, axis=None):
        """
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
        """
        # This annoying conditional is necessary despite torch documentation
        # suggesting the contrary:
        # https://pytorch.org/docs/stable/tensors.html
        #
        # ctrl+f `mean` for the incorrect default signature that raises:
        # RuntimeError: Please look up dimensions by name, got: name = None.
        #
        # Update: This reference has been removed from the documentation,
        # which suggests this conditional is likely here to stay.
        out = x.detach().clone()
        bound = torch.tensor(bound, dtype=x.dtype, device=x.device)
        if axis is None:
            upper = out.amax()
            lower = out.amin()
            unew = torch.minimum(bound[-1], out.amax())
            lnew = torch.maximum(bound[0], out.amin())
            out -= out.mean()
            out /= ((upper - lower) / (unew - lnew))
            out += (lnew - out.amin())
        else:
            upper = out.amax(axis)
            lower = out.amin(axis)
            unew = torch.minimum(bound[-1], out.amax(axis))
            lnew = torch.maximum(bound[0], out.amin(axis))
            out -= out.mean(axis)
            out /= ((upper - lower) / (unew - lnew))
            out += (lnew - out.amin(axis))
        return out


class _Domain(torch.nn.Module):
    """
    Functional domain mapper.

    Map a tensor to either its image or its preimage under some function in a
    safe manner that handles out-of-domain entries to avoid nan and error
    values. Used as a superclass for parameter-constraining mappers.

    Parameters/Attributes
    ---------------------
    loc : float (default 0)
        Translation applied to the image under the specified function. Scaling
        and translation are applied after the base function.
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
    """
    def __init__(self, loc=0, scale=1, bound=None, limits=None, handler=None):
        super(_Domain, self).__init__()
        self.handler = handler or Clip()
        bound = bound or [-float('inf'), float('inf')]
        limits = limits or [-float('inf'), float('inf')]
        self.bound = bound
        self.limits = limits
        self.loc = loc
        self.scale = scale
        self.signature = {}

    def preimage(self, x):
        """
        Map a tensor to its preimage under the transformation. Any values
        outside the transformation's range are first handled.
        """
        bound = torch.tensor(self.bound, dtype=x.dtype, device=x.device)
        limits = torch.tensor(self.limits, dtype=x.dtype, device=x.device)
        x = self.handler.apply(x, bound)
        i = self.preimage_map((x - self.loc) / self.scale)
        i = self.handler.apply(i, limits)
        return i

    def image(self, x):
        """
        Map a tensor to its image under the transformation.
        """
        return self.scale * self.image_map(x) + self.loc

    def preimage_dim(self, dim):
        """
        Determine the dimension of a tensor's preimage under the
        transformation. Input can be either a tensor or an iterable describing
        its original dimension.
        """
        if isinstance(dim, torch.Tensor):
            dim = dim.shape
        dim = list(dim)
        for k, (v, _) in self.signature.items():
            dim[k] = v(dim[k])
        return torch.Size(dim)

    def image_dim(self, dim):
        """
        Determine the dimension of a tensor's image under the transformation.
        Input can be either a tensor or an iterable describing its original
        dimension.
        """
        if isinstance(dim, torch.Tensor):
            dim = dim.shape
        dim = list(dim)
        for k, (_, v) in self.signature.items():
            dim[k] = v(dim[k])
        return torch.Size(dim)

    def test(self, x):
        """
        Evaluate whether each entry in a tensor falls within bounds.
        """
        return self.handler.test(x, self.bound)

    def handle_ood(self, x):
        """
        Apply an out-of-domain handler to ensure that all tensor entries are
        within bounds.
        """
        return self.handler.apply(x)

    def extra_repr(self):
        s = []
        if self.scale != 1:
            s += [f'scale={self.scale}']
        if not all([math.isinf(i) for i in self.bound]):
            s += [f'bound=({self.bound[0]}, {self.bound[1]})']
            s += [f'handler={self.handler.__repr__()}']
        return ', '.join(s)


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
    """
    def preimage(self, x):
        """
        Map the amplitude of a complex-valued tensor to its preimage under the
        transformation. Any values outside the transformation's range are first
        handled.
        """
        ampl, phase = complex_decompose(x)
        ampl = super(_PhaseAmplitudeDomain, self).preimage(ampl)
        return complex_recompose(ampl, phase)

    def image(self, x):
        """
        Map the amplitude of a complex-valued a tensor to its image under the
        transformation.
        """
        ampl, phase = complex_decompose(x)
        ampl = super(_PhaseAmplitudeDomain, self).image(ampl)
        return complex_recompose(ampl, phase)


class Identity(_Domain):
    """
    Identity domain mapper.
    """
    def preimage(self, x):
        """
        Returns exactly the unmodified input.
        """
        return x

    def image(self, x):
        """
        Returns exactly the unmodified input.
        """
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
    """
    def __init__(self, scale=1):
        super(Linear, self).__init__()
        self.scale = scale

    def preimage(self, x):
        """
        Map a tensor to its preimage under the linear transformation.
        """
        return x / self.scale

    def image(self, x):
        """
        Map a tensor to its image under the linear transformation.
        """
        return self.scale * x


class Affine(_Domain):
    """
    Affine domain mapper.

    Parameters/Attributes
    ---------------------
    loc : float (default 0)
        Intercept of the affine map. Offset/translation applied to the image.
    scale : float (default 1)
        Slope of the affine map. Scale factor applied to the image. The
        multiplicative inverse (reciprocal) is applied before the preimage is
        computed.
    """
    def __init__(self, loc=0, scale=1):
        super(Linear, self).__init__()
        self.loc = loc
        self.scale = scale

    def preimage(self, x):
        """
        Map a tensor to its preimage under the linear transformation.
        """
        return (x - self.loc) / self.scale

    def image(self, x):
        """
        Map a tensor to its image under the linear transformation.
        """
        return self.scale * x + self.loc
