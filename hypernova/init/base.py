# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Initialiser
~~~~~~~~~~~
Base initialiser for a module.
"""
import torch


def uniform_init_(tensor, min=0, max=1):
    """
    Populate a tensor with values uniformly sampled i.i.d. from a specified
    interval. The tensor is updated in place.

    TODO: This should be implemented as a specific case of an init from
          distribution.

    Parameters
    ----------
    tensor : Tensor
        Tensor to populate or initialise from the specified distribution.
    min : float
        Lower bound of the interval from which the tensor's elements are
        sampled.
    max : float
        Upper bound of the interval from which the tensor's elements are
        sampled.
    """
    val = torch.rand_like(tensor) * (max - min) + min
    tensor[:] = val


class DomainInitialiser(object):
    """
    Initialiser for a tensor whose values are the preimage of some function.

    For example, a layer can internally store a "preweight" that is passed
    through a logistic function to produce the actual weight seen by data in
    the forward pass. This constrains the actual weight to the interval (0, 1)
    and makes the unconstrained preweight the learnable parameter. We might
    often wish to initialise the actual weight from some distribution rather
    than initialising the preweight; this class provides a convenient way to
    do so.

    A `DomainInitialiser` is callable with a single required argument: a
    tensor to be initialised following the specified initialisation scheme.

    Parameters
    ----------
    init : callable
        A python callable that takes as its single required parameter the
        tensor that is to be initialised; the callable should, when called,
        initialise the tensor in place. Callables with additional arguments
        can be constrained using `partial` from `functools` or an appropriate
        lambda function. If no `init` is explicitly specified,
        `DomainInitialiser` defaults to a uniform initialisation in the
        interval (0, 1).
    domain : Domain object
        A representation of the function used to map between the learnable
        preweight and the weight "seen" by the data. It must have a `preimage`
        method that maps values in the weight domain to their preimage under
        the function: the corresponding values in the preweight domain.
        Examples are provided in `functional.domain`. If no `domain` is
        explicitly specified, `DomainInitialiser` defaults to identity
        (preweight and weight are the same).
    """
    def __init__(self, init=None, domain=None):
        self.init = init or uniform_init_
        self.domain = domain or Identity()

    def __call__(self, tensor):
        rg = tensor.requires_grad
        tensor.requires_grad = False
        self.init(tensor)
        tensor[:] = self.domain.preimage(tensor)
        tensor.requires_grad = rg


class BaseInitialiser(DomainInitialiser):
    """
    Basic initialiser class. This class mostly exists to be subclassed.

    Parameters
    ----------
    init : callable
        A python callable that takes as its single required parameter the
        tensor that is to be initialised; the callable should, when called,
        initialise the tensor in place. Callables with additional arguments
        can be constrained using `partial` from `functools` or an appropriate
        lambda function. If no `init` is explicitly specified,
        `BaseInitialiser` defaults to a uniform initialisation in the
        interval (0, 1).
    """
    def __init__(self, init=None):
        super(BaseInitialiser, self).__init__(init=init, domain=None)
