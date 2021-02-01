# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Shorthand
~~~~~~~~~
Shorthand expressions and parser for the confound modelling subsystem.
"""
import re
from itertools import chain


class ShorthandFilter(object):
    def __init__(self, pattern):
        self.pattern = re.compile(pattern)


class Shorthand(object):
    def __init__(self, rules=None, regex=None, filters=None, transforms=None):
        self.rules = rules
        self.regex = regex
        self.filters = filters
        self.transforms = transforms

    def expand(self, model_formula, variables, metadata=None):
        for k, filt in self.filters.items():
            params = re.search(k, model_formula)
            if params is None: continue
            v = filt(metadata, **params.groupdict())
            model_formula = re.sub(k, v, model_formula)
        for k, v in self.regex.items():
            v = self.find_matching_variables(v, variables)
            model_formula = re.sub(k, v, model_formula)
        for k, v in self.rules.items():
            model_formula = re.sub(k, v, model_formula)

        formula_variables = self.get_formula_variables(model_formula)
        others = ' + '.join(set(variables) - set(formula_variables))
        model_formula = re.sub('others', others, model_formula)
        return model_formula

    def find_matching_variables(self, regex, variables):
        matches = re.compile(regex)
        matches = ' + '.join([v for v in variables if matches.match(v)])
        return matches

    def get_formula_variables(self, model_formula):
        symbols_to_clear = [[m.regex for m in t.matches]
                            for t in self.transforms]
        symbols_to_clear = list(chain(*symbols_to_clear))
        for symbol in symbols_to_clear:
            model_formula = re.sub(symbol, '', model_formula)
        variables = model_formula.split('+')
        return variables

    def __call__(self, model_formula, variables, metadata=None):
        return self.expand(model_formula, variables, metadata)
