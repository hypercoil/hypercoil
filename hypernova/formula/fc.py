# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Functional connectivity
~~~~~~~~~~~~~~~~~~~~~~~
Initialisations of base data classes for modelling functional connectivity.
"""
import pandas as pd
from functools import reduce
from .coltransforms import ColumnTransform, OrderedTransform, MatchOnly
from .model import ModelSpec
from .shorthand import Shorthand, ShorthandFilter
from .utils import diff_nanpad, match_metadata, numbered_string


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
        transforms = [
            DerivativeTransform(),
            PowerTransform(),
            ThreshBinTransform(),
            UThreshBinTransform(),
            UnionTransform(),
            IntersectionTransform()
        ]
        shorthand, shorthand_re, shorthand_filters = fc_shorthand()
        super(FCShorthand, self).__init__(
            rules=shorthand,
            regex=shorthand_re,
            filters=shorthand_filters,
            transforms=transforms
        )


class PowerTransform(OrderedTransform):
    """
    Column transform that computes exponential expansions. The nth-order
    transform returns the input data raised to the nth power.

    Formula specifications
    ----------------------
    * (variable)^6 for the 6th power
    * (variable)^^6 for all powers up to the 6th
    * (variable)^4-6 for the 4th through 6th powers
    * 1 must be included in the powers range for the original term to be
      returned when exponential terms are computed.
    """
    def __init__(self):
        super(PowerTransform, self).__init__(
            transform=lambda data, order: data.values ** order,
            all=r'^\((?P<child0>.*)\)\^\^(?P<order>[0-9]+)$',
            select=r'^\((?P<child0>.*)\)\^(?P<order>[0-9]+[\-]?[0-9]*)$',
            first=1,
            name='power',
            identity=1
        )


class DerivativeTransform(OrderedTransform):
    """
    Column transform that computes temporal derivatives as backward
    differences. The nth-order transform returns the nth-order backward
    difference of the input data, padding the output with NaNs so that its
    shape matches that of the input.

    Formula specifications
    ----------------------
    * d6(variable) for the 6th temporal derivative
    * dd6(variable) for all temporal derivatives up to the 6th
    * d4-6(variable) for the 4th through 6th temporal derivatives
    * 0 must be included in the temporal derivative range for the original
      term to be returned when temporal derivatives are computed.
    """
    def __init__(self):
        super(DerivativeTransform, self).__init__(
            transform=lambda data, order: diff_nanpad(data, n=order, axis=0),
            all=r'^dd(?P<order>[0-9]+)\((?P<child0>.*)\)$',
            select=r'^d(?P<order>[0-9]+[\-]?[0-9]*)\((?P<child0>.*)\)$',
            first=0,
            name='derivative',
            identity=0
        )


class ThreshBinTransform(ColumnTransform):
    """
    Column transform that thresholds each variable and binarises the result,
    thus returning a Boolean-valued variable whose observations indicate
    whether the corresponding observations of the original variable survived
    the threshold. Note that any missing or NaN observation is mapped to False.

    Formula specifications
    ----------------------
    * thr0.5(variable) for an indicator of whether each observation is greater
      than 0.5

    See also
    --------
    `UThreshBinTransform`: create an indicator variable specifying whether each
    observation is under some threshold. TODO: In the future, thresholding
    transformations will be replaced by a single transformation that separately
    handles lt, le, gt, ge, ne, eq cases.
    """
    def __init__(self):
        regex = r'^thr(?P<thresh>[0-9]+[\.]?[0-9]*)\((?P<child0>.*)\)$'
        transform = lambda data, thresh: data.values > thresh
        typedict = {'thresh': float}
        matches = [MatchOnly(regex=regex, typedict=typedict)]
        super(ThreshBinTransform, self).__init__(
            transform=transform,
            matches=matches,
            name='threshbin'
        )

    def argform(self, **args):
        return args['thresh']


class UThreshBinTransform(ColumnTransform):
    """
    Column transform that upper-thresholds each variable and binarises the
    result, thus returning a Boolean-valued variable whose observations
    indicate whether the corresponding observations of the original variable
    survived the upper threshold. Note that any missing or NaN observation is
    mapped to False.

    Formula specifications
    ----------------------
    * uthr0.5(variable) for an indicator of whether each observation is less
      than 0.5

    See also
    --------
    `ThreshBinTransform`: create an indicator variable specifying whether each
    observation is above some threshold. TODO: In the future, thresholding
    transformations will be replaced by a single transformation that separately
    handles lt, le, gt, ge, ne, eq cases.
    """
    def __init__(self):
        regex = r'^uthr(?P<thresh>[0-9]+[\.]?[0-9]*)\((?P<child0>.*)\)$'
        transform = lambda data, thresh: data.values < thresh
        typedict = {'thresh': float}
        matches = [MatchOnly(regex=regex, typedict=typedict)]
        super(UThreshBinTransform, self).__init__(
            transform=transform,
            matches=matches,
            name='uthreshbin'
        )

    def argform(self, **args):
        return args['thresh']


class UnionTransform(ColumnTransform):
    """
    Column transform that replaces all input variables with a single Boolean-
    valued variable representing their logical union. Input variables should be
    Boolean-valued.

    Formula specifications
    ----------------------
    * or(variable0 + variable1) for the elementwise union of Boolean-valued
      observations in variable0 and variable1
    * or(thr1(variable) + uthr0(variable)) for an indicator specifying if the
      value of `variable` is in the complement of (0, 1).

    See also
    --------
    `IntersectionTransform` : logical intersection.
    """
    def __init__(self):
        regex = r'^or\((?P<child0>.*)\)'
        transform = lambda values: reduce((lambda x, y: x | y), values.T)
        matches = [MatchOnly(regex=regex)]
        super(UnionTransform, self).__init__(
            transform=transform,
            matches=matches,
            name='union'
        )

    def __call__(self, children, **args):
        selected = children[0]
        vars = '_OR_'.join(selected.columns)
        return pd.DataFrame(
            data=self.transform(selected.values),
            columns=[f'union_{vars}']
        )


class IntersectionTransform(ColumnTransform):
    """
    Column transform that replaces all input variables with a single Boolean-
    valued variable representing their logical intersection. Input variables
    should be Boolean-valued.

    Formula specifications
    ----------------------
    * and(variable0 + variable1) for the elementwise intersection of Boolean-
      valued observations in variable0 and variable1
    * and(thr0(variable) + uthr1(variable)) for an indicator specifying if the
      value of `variable` is in (0, 1).

    See also
    --------
    `UnionTransform` : logical union.
    """
    def __init__(self):
        regex = r'^and\((?P<child0>.*)\)'
        transform = lambda values: reduce((lambda x, y: x & y), values.T)
        matches = [MatchOnly(regex=regex)]
        super(IntersectionTransform, self).__init__(
            transform=transform,
            matches=matches,
            name='intersection'
        )

    def __call__(self, children, **args):
        selected = children[0]
        vars = '_AND_'.join(selected.columns)
        return pd.DataFrame(
            data=self.transform(selected.values),
            columns=[f'intersection_{vars}']
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
            transforms=[
                PowerTransform(),
                DerivativeTransform(),
                ThreshBinTransform(),
                UThreshBinTransform(),
                UnionTransform(),
                IntersectionTransform()
            ]
        )
