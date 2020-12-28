# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Functional connectivity
~~~~~~~~~~~~~~~~~~~~~~~
Initialisations of base data classes for modelling functional connectivity.
"""
from .coltransforms import ColumnTransform
from .model import ModelSpec
from .shorthand import Shorthand, ShorthandFilter
from .utils import diff_nanpad, match_metadata, numbered_string


def fc_shorthand():
    rules = {
        'wm': 'white_matter',
        'gsr': 'global_signal',
        'rps': 'trans_x + trans_y + trans_z + rot_x + rot_y + rot_z',
        'fd': 'framewise_displacement'
    }
    regex = {
        'acc': '^a_comp_cor_[0-9]+',
        'tcc': '^t_comp_cor_[0-9]+',
        'dv': '^std_dvars$',
        'dvall': '.*dvars$',
        'nss': '^non_steady_state_outlier[0-9]+',
        'spikes': '^motion_outlier[0-9]+'
    }
    filters = {
        'acc\<n=(?P<n>[0-9]+)(,)?(\s)?(mask=(?P<mask>[A-Za-z\+]*))?\>':
            FirstN('^a_comp_cor_[0-9]+'),
        'acc\<v=(?P<v>[0-9\.]+)(,)?(\s)?(mask=(?P<mask>[A-Za-z\+]*))?\>':
            CumulVar('^a_comp_cor_[0-9]+'),
        'tcc\<n=(?P<n>[0-9]+)\>': FirstN('^t_comp_cor_[0-9]+'),
        'tcc\<v=(?P<v>[0-9\.]+)\>': CumulVar('^t_comp_cor_[0-9]+'),
        'aroma': NoiseComponents('^aroma_motion_[0-9]+')
    }
    return rules, regex, filters


class FirstN(ShorthandFilter):
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
    def __call__(self, metadata):
        out = []
        matches = match_metadata(self.pattern, metadata)
        for m in matches:
            item = metadata.get(m)
            if item['MotionNoise']: out += [m]
        return ' + '.join(out)


class FCShorthand(Shorthand):
    def __init__(self):
        transforms = [
            DerivativeTransform(),
            PowerTransform()
        ]
        shorthand, shorthand_re, shorthand_filters = fc_shorthand()
        super(FCShorthand, self).__init__(
            rules=shorthand,
            regex=shorthand_re,
            filters=shorthand_filters,
            transforms=transforms
        )


class PowerTransform(ColumnTransform):
    """
    Compute exponential expansions.

    Parameters
    ----------
    order: set(int)
        A set of exponential terms to include. For instance, {1, 2}
        indicates that the first and second powers should be added.
        To retain the original terms, 1 *must* be included in the list.
    variables: list(str)
        List of variables for which exponential terms should be computed.
    data: pandas DataFrame object
        Table of values of all observations of all variables.

    Returns
    -------
    variables_exp: list
        A list of variables to include in the final data frame after adding
        the specified exponential terms.
    data_exp: pandas DataFrame object
        Table of values of all observations of all variables, including any
        specified exponential terms.

    """
    def __init__(self):
        super(PowerTransform, self).__init__(
            transform=lambda data, order: data.values ** order,
            all=r'\^\^([0-9]+)$',
            select=r'\^([0-9]+[\-]?[0-9]*)$',
            first=1,
            name='power',
            identity=1
        )


class DerivativeTransform(ColumnTransform):
    """
    Compute temporal derivative terms by the method of backwards differences.

    Parameters
    ----------
    order: set(int)
        A set of temporal derivative terms to include. For instance, {1, 2}
        indicates that the first and second derivative terms should be added.
        To retain the original terms, 0 *must* be included in the set.
    variables: list(str)
        List of variables for which temporal derivative terms should be
        computed.
    data: pandas DataFrame object
        Table of values of all observations of all variables.

    Returns
    -------
    variables_deriv: list
        A list of variables to include in the final data frame after adding
        the specified derivative terms.
    data_deriv: pandas DataFrame object
        Table of values of all observations of all variables, including any
        specified derivative terms.

    """
    def __init__(self):
        super(DerivativeTransform, self).__init__(
            transform=lambda data, order: diff_nanpad(data, n=order, axis=0),
            all=r'^dd([0-9]+)',
            select=r'^d([0-9]+[\-]?[0-9]*)',
            first=0,
            name='derivative',
            identity=0
        )


class FCConfoundModelSpec(ModelSpec):
    """
    model_formula: str
        Expression for the model formula, e.g.
        '(a + b)^^2 + dd1(c + (d + e)^3) + f'
        Note that any expressions to be expanded *must* be in parentheses,
        even if they include only a single variable (e.g., (x)^2, not x^2).
    parent_data: pandas DataFrame
        A tabulation of all values usable in the model formula. Each additive
        term in `model_formula` should correspond either to a variable in this
        data frame or to instructions for operating on a variable (for
        instance, computing temporal derivatives or exponential terms).

    Options
    -------
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
    """
    def __init__(self, spec, name=None):
        name = name or 'confounds'
        super(FCConfoundModelSpec, self).__init__(
            spec=spec,
            name=name,
            shorthand=FCShorthand(),
            transforms=[
                PowerTransform(),
                DerivativeTransform()
            ]
        )
