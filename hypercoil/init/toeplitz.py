# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Initialise parameters as a stack of Toeplitz-structured banded matrices.
"""
import jax
from typing import Optional, Tuple, Type
from .base import BaseInitialiser, MappedInitialiser
from .mapparam import MappedParameter
from ..functional import toeplitz
from ..functional.utils import PyTree, Tensor


class ToeplitzInitialiser(MappedInitialiser):
    """
    Banded matrix initialisation.

    Initialise a tensor as a stack of banded matrices with Toeplitz structure.

    See :func:`toeplitz` for argument details.
    """

    c : Tensor
    r : Optional[Tensor] = None
    fill_value : float = 0.

    def __init__(
        self, c, r=None, fill_value=0,
        mapper: Optional[Type[MappedParameter]] = None):
        self.c = c
        self.r = r
        self.fill_value = fill_value
        super().__init__(mapper=mapper)

    def _init(
        self,
        shape: Tuple[int],
        key: Optional[jax.random.PRNGKey] = None,
    ) -> Tensor:
        return toeplitz(
            c=self.c, r=self.r, fill_value=self.fill_value, shape=shape)

    @classmethod
    def init(
        cls,
        model: PyTree,
        *,
        mapper: Optional[Type[MappedParameter]] = None,
        c: Tensor,
        r: Optional[Tensor] = None,
        fill_value: float = 0.,
        param_name: str = "weight",
        key: Optional[jax.random.PRNGKey] = None,
    ):
        init = cls(mapper=mapper, c=c, r=r, fill_value=fill_value)
        return super()._init_impl(
            init=init, model=model, param_name=param_name, key=key,
        )


class ToeplitzInit(BaseInitialiser):
    def __init__(self):
        raise NotImplementedError()

def toeplitz_init_(tensor, c, r=None, fill_value=0, domain=None):
    raise NotImplementedError()
