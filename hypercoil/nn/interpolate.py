# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Modules for performing interpolation.
"""
import jax
import equinox as eqx
from typing import Optional
from ..engine import Tensor
from ..functional import (
    spectral_interpolate,
    linear_interpolate,
    hybrid_interpolate
)


class SpectralInterpolate(eqx.Module):
    """
    :doc:`Spectral interpolation <hypercoil.functional.interpolate.spectral_interpolate>`
    module.
    """
    oversampling_frequency: float = 8
    maximum_frequency: float = 1
    sampling_period: float = 1
    thresh: float = 0

    def __call__(
        self,
        input: Tensor,
        mask: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        return spectral_interpolate(
            data=input,
            mask=mask,
            oversampling_frequency=self.oversampling_frequency,
            maximum_frequency=self.maximum_frequency,
            sampling_period=self.sampling_period,
            thresh=self.thresh
        )


class LinearInterpolate(eqx.Module):
    """
    :doc:`Linear interpolation <hypercoil.functional.interpolate.linear_interpolate>`
    module.
    """
    def __call__(
        self,
        input: Tensor,
        mask: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        return linear_interpolate(
            data=input,
            mask=mask,
        )


class HybridInterpolate(eqx.Module):
    """
    :doc:`Hybrid interpolation <hypercoil.functional.interpolate.hybrid_interpolate>`
    module.
    """
    max_consecutive_linear: int = 3
    oversampling_frequency: float = 8
    maximum_frequency: float = 1
    frequency_thresh: float = 0.3

    def __call__(
        self,
        input: Tensor,
        mask: Tensor,
        *,
        key: Optional['jax.random.PRNGKey'] = None,
    ) -> Tensor:
        return hybrid_interpolate(
            data=input,
            mask=mask,
            max_consecutive_linear=self.max_consecutive_linear,
            oversampling_frequency=self.oversampling_frequency,
            maximum_frequency=self.maximum_frequency,
            frequency_thresh=self.frequency_thresh
        )
