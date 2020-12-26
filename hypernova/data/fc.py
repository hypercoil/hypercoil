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
from .utils import diff_nanpad


shorthand = {
    'wm': 'white_matter',
    'gsr': 'global_signal',
    'rps': 'trans_x + trans_y + trans_z + rot_x + rot_y + rot_z',
    'fd': 'framewise_displacement'
}
shorthand_re = {
    'acc': '^a_comp_cor_[0-9]+',
    'tcc': '^t_comp_cor_[0-9]+',
    'dv': '^std_dvars$',
    'dvall': '.*dvars$',
    'nss': '^non_steady_state_outlier[0-9]+',
    'spikes': '^motion_outlier[0-9]+'
}
shorthand_filters = {
    'acc\<n=(?P<n>[0-9]+)\>': FirstN('^a_comp_cor_[0-9]+'),
    'acc\<v=(?P<v>[0-9\.]+)\>': CumulVar('^a_comp_cor_[0-9]+'),
    'tcc\<n=(?P<n>[0-9]+)\>': FirstN('^t_comp_cor_[0-9]+'),
    'tcc\<v=(?P<v>[0-9\.]+)\>': CumulVar('^t_comp_cor_[0-9]+'),
}


class FirstN(ShorthandFilter):
    def __call__(self, metadata, n):
        n = int(n)
        matches = list(filter(self.pattern.match, metadata.keys()))
        matches.sort(key=numbered_string)
        return ' + '.join(matches[:n])


class CumulVar(ShorthandFilter):
    def __call__(self, metadata, v):
        v = float(v)
        if v > 1: v /= 100
        out = []
        matches = list(filter(self.pattern.match, metadata.keys()))
        for m in matches:
            item = metadata.get(m)
            if not item: break
            out += [m]
            if item['CumulativeVarianceExplained'] > v: break
        return ' + '.join(out)


class FCShorthand(Shorthand):
    def __init__(self):
        transforms = [
            DerivativeTransform(),
            PowerTransform()
        ]
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
            transform=lambda data, order: data ** order,
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
