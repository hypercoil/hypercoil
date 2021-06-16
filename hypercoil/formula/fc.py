# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Functional connectivity
~~~~~~~~~~~~~~~~~~~~~~~
Initialisations of base data classes for modelling functional connectivity.
"""
from .model import ModelSpec
from .shorthand import Shorthand, ShorthandFilter
from .transforms import (
    DerivativeTransform,
    PowerTransform,
    ThreshIndicatorTransform,
    UnionTransform,
    IntersectionTransform,
    NegationTransform
)
from .utils import match_metadata, numbered_string


def fc_shorthand():
    """
    Return the shorthand rules for functional connectivity.
    """
    rules = {
        'wm': 'white_matter',
        'gsr': 'global_signal',
        'rps': 'trans_x + trans_y + trans_z + rot_x + rot_y + rot_z',
        'fd': 'framewise_displacement'
    }
    regex = {
        'acc': r'^a_comp_cor_[0-9]+',
        'tcc': r'^t_comp_cor_[0-9]+',
        'dv': r'^std_dvars$',
        'dvall': r'.*dvars$',
        'nss': r'^non_steady_state_outlier[0-9]+',
        'spikes': r'^motion_outlier[0-9]+'
    }
    filters = {
        r'acc\<n=(?P<n>[0-9]+)(,)?(\s)?(mask=(?P<mask>[A-Za-z\+]*))?\>':
            FirstN(r'^a_comp_cor_[0-9]+'),
        r'acc\<v=(?P<v>[0-9\.]+)(,)?(\s)?(mask=(?P<mask>[A-Za-z\+]*))?\>':
            CumulVar(r'^a_comp_cor_[0-9]+'),
        r'tcc\<n=(?P<n>[0-9]+)\>': FirstN(r'^t_comp_cor_[0-9]+'),
        r'tcc\<v=(?P<v>[0-9\.]+)\>': CumulVar(r'^t_comp_cor_[0-9]+'),
        r'aroma': NoiseComponents(r'^aroma_motion_[0-9]+')
    }
    return rules, regex, filters


def fc_transforms():
    return [
        DerivativeTransform(),
        PowerTransform(),
        ThreshIndicatorTransform(),
        UnionTransform(),
        IntersectionTransform(),
        NegationTransform(),
    ]


class FirstN(ShorthandFilter):
    """
    Return the first n numbered strings matching a pattern (e.g., the first n
    strings beginning with `a_comp_cor`).
    """
    def __call__(self, metadata, n, mask=None):
        n = int(n)
        matches = match_metadata(self.pattern, metadata)
        matches.sort(key=numbered_string)
        if mask:
            mmsk = []
            masks = mask.split('+')
            for msk in masks:
                filt = [m for m in matches if metadata[m].get('Mask') == msk]
                mmsk += filt[:n]
            return ' + '.join(mmsk)
        return ' + '.join(matches[:n])


class CumulVar(ShorthandFilter):
    """
    Return the first set of numbered strings sufficient to cumulatively explain
    some fraction v of variance (as specified in the metadata).
    """
    def __call__(self, metadata, v, mask=None):
        v = float(v)
        if v > 1: v /= 100
        out = []
        matches = match_metadata(self.pattern, metadata)
        if mask:
            masks = mask.split('+')
            matches = [m for m in matches if metadata[m].get('Mask') in masks]
        done = False
        for m in matches:
            item = metadata.get(m)
            if not item: break
            if done:
                done = item['CumulativeVarianceExplained'] > v
                if done: continue
            done = item['CumulativeVarianceExplained'] > v
            out += [m]
        return ' + '.join(out)


class NoiseComponents(ShorthandFilter):
    """
    Return components flagged as noise in the metadata.
    """
    def __call__(self, metadata):
        out = []
        matches = match_metadata(self.pattern, metadata)
        for m in matches:
            item = metadata.get(m)
            if item['MotionNoise']: out += [m]
        return ' + '.join(out)


class FCShorthand(Shorthand):
    """Shorthand rules for functional connectivity confound models."""
    def __init__(self):
        transforms = fc_transforms()
        shorthand, shorthand_re, shorthand_filters = fc_shorthand()
        super(FCShorthand, self).__init__(
            rules=shorthand,
            regex=shorthand_re,
            filters=shorthand_filters,
            transforms=transforms
        )


class FCConfoundModelSpec(ModelSpec):
    """
    Model specification for confound models used to denoise data before
    functional connectivity analysis.

    Parameters
    ----------
    spec : str
        Expression for the model formula, e.g.
        '(a + b)^^2 + dd1(c + (d + e)^3) + f'
        Note that any expressions to be expanded *must* be in parentheses,
        even if they include only a single variable (e.g., (x)^2, not x^2).
    name : str
        Name of the model. If none is provided, then the string 'confounds'
        will be used by default. Note that this will lead to a hash collision
        if multiple models are entered by name into a hash table.

    Specification
    -------------
    Variables and transformation instructions are written additively; the
    model is specified as their sum. Each term in the sum is either a variable
    or a transformation of another variable or sum of transformations/
    variables. Permitted transformations are temporal derivatives and powers.

    Temporal derivative options:
    * d6(variable) for the 6th temporal derivative
    * dd6(variable) for all temporal derivatives up to the 6th
    * d4-6(variable) for the 4th through 6th temporal derivatives
    * 0 must be included in the temporal derivative range for the original
      term to be returned when temporal derivatives are computed.

    Exponential options:
    * (variable)^6 for the 6th power
    * (variable)^^6 for all powers up to the 6th
    * (variable)^4-6 for the 4th through 6th powers
    * 1 must be included in the powers range for the original term to be
      returned when exponential terms are computed.

    Temporal derivatives and exponential terms are computed for all terms
    in the grouping symbols that they adjoin.

    Example specs
    -------------
    (dd1(rps + wm + csf + gsr))^^2
        36-parameter model: 6 realignment parameters (3 translational, 3
        rotational), mean white matter and cerebrospinal fluid time series,
        mean global time series, temporal derivatives of all time series, and
        squares of the original time series and their derivatives.
    aroma + wm + csf
        ICA-AROMA noise time series concatenated with mean white matter and
        cerebrospinal fluid time series.
    acc(v=50, mask=CSF+WM) + rps
        Anatomical CompCor time series sufficient to explain 50 percent of
        variance in the white matter and CSF masks together with the 6
        realignment parameters.
    acc(n=6, mask=combined) + rps
        First 6 anatomical CompCor time series from the combined WM/CSF mask
        and the 6 realignment parameters.
    """
    def __init__(self, spec, name=None):
        name = name or spec
        super(FCConfoundModelSpec, self).__init__(
            spec=spec,
            name=name,
            shorthand=FCShorthand(),
            transforms=fc_transforms()
        )
