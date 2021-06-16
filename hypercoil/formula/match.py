# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Match rules
~~~~~~~~~~~
Rules for transforming text that matches a regular expression.
"""
import re


class MatchRule(object):
    """
    A rule for transforming text that matches a regular expression.

    The regular expression should capture fields of interest in named groups.
    The match rule first casts each field in the group dictionary to any type
    specified in its `typedict`. It then applies the function specified as its
    `rule` to the type-cast group dictionary. The function is applied in-place.

    Parameters
    ----------
    regex : str
        String representation of the regular expression. It should contain
        named capturing groups (patterned as `(?P<name>pattern_to_match)`)
        corresponding to each variable that should be extracted from strings
        at match time.
    rule : callable(dict)
        In-place function that transforms the type-cast group dictionary.
    typedict : dict or None (default None)
        Dictionary specifying the type that each matched group should be cast
        to. Keys are the same as group names, and values are types. If any
        group is not included, it will be left as a string (although the
        `rule` might later convert it). If this is None, then no type casting
        will be performed using the `typedict`.
    """
    def __init__(self, regex, rule, typedict=None):
        self.regex = re.compile(regex)
        self.rule = rule
        self.typedict = typedict or {}

    def _cast_all(self, groupdict):
        """Use the `typedict` to cast parsed groups with matching names."""
        for k, v in groupdict.items():
            t = self.typedict.get(k)
            if t:
                groupdict[k] = t(v)
        return groupdict

    def __call__(self, expr):
        """
        Parse an expression: check whether it contains a match to the `regex`
        and, if so, apply the `typedict` and `rule` to all fields.
        """
        m = re.match(self.regex, expr)
        if m:
            parsed = m.groupdict()
            self.rule(self._cast_all(parsed))
            return parsed


class MatchOnly(MatchRule):
    """
    A simple `MatchRule` that matches a regular expression without applying
    any rules to the matched groups.

    Parameters
    ----------
    regex : str
        String representation of the regular expression. It should contain
        named capturing groups (patterned as `(?P<name>pattern_to_match)`)
        corresponding to each variable that should be extracted from strings
        at match time.
    typedict : dict or None (default None)
        Dictionary specifying the type that each matched group should be cast
        to. Keys are the same as group names, and values are types. If any
        group is not included, it will be left as a string (although the
        `rule` might later convert it). If this is None, then no type casting
        will be performed using the `typedict`.
    """
    def __init__(self, regex, typedict=None):
        rule = lambda x: x
        super(MatchOnly, self).__init__(
            regex=regex, rule=rule, typedict=typedict)


class MatchAndCompare(MatchRule):
    """
    A `MatchRule` that maps the `compare` field to a Boolean-valued function
    to a function that returns true if the string representation of the
    comparison operator stored in the `compare` field is satisfied.

    Parameters
    ----------
    regex : str
        String representing the regular expression indicating variables to
        be compared, the comparison operator, and the reference point. The
        regular expression should contain as subexpressions:
        - r'^(?P<child0>[^\>\<\=\!]*)' for variables to compare;
        - r'(?P<compare>[\>\<\=\!]+) *' for the comparison operator;
        - r'(?P<thresh>[0-9]+[\.]?[0-9]*)\]' for the reference threshold
    """
    def __init__(self, regex):
        rule = lambda x: x.update(compare=self._comparison(x['compare']))
        typedict = {
            'thresh': float
        }
        super(MatchAndCompare, self).__init__(
            regex=regex,
            rule=rule,
            typedict=typedict
        )

    def _comparison(self, compare):
        """
        Convert a string representation of a comparison operator into a
        function that returns true if its input satisfies the operator in
        relation to some threshold.
        """
        if compare == '>':
            return lambda x, thresh: x > thresh
        elif compare == '<':
            return lambda x, thresh: x < thresh
        elif compare == '>=':
            return lambda x, thresh: x >= thresh
        elif compare == '<=':
            return lambda x, thresh: x <= thresh
        elif compare == '=' or compare == '==':
            return lambda x, thresh: x == thresh
        elif compare == '!=' or compare == '~=':
            return lambda x, thresh: x != thresh
        else:
            raise ValueError(f'Invalid comparison string: {compare}')


class AllOrders(MatchRule):
    """
    A `MatchRule` that transforms the `order` into a set of integers from
    `first` to the match, inclusive. Used to parse orders for an
    `OrderedTransform`.

    Parameters
    ----------
    regex : str
        String representing the regular expression that indicates that all
        orders of the transform beginning from `first` until a specified order
        are to be applied and concatenated. The regular expression must
        contain the symbols used to represent the transform as well as the
        exact parenthetic string `(?P<order>[0-9]+)` in the position where the
        transform order is specified. Alongside the `select` argument, this is
        the regular expression that will be matched in any Expressions using
        this transform. Consult `PowerTransform` and `DerivativeTransform` for
        examples.
    first : int
        Integer denoting the smallest possible transformation order for an
        ordered transform. For instance, a transformation representing
        derivatives could begin with the zeroth derivative, corresponding to
        identity. (Including an identity transformation supports easier
        notation if the transform is to be used in a model specification.)
    """
    def __init__(self, regex, first=0):
        self.first = first
        rule = lambda x: x.update(order=set(range(self.first, x['order'] + 1)))
        typedict = {
            'order': int
        }
        super(AllOrders, self).__init__(
            regex=regex,
            rule=rule,
            typedict=typedict
        )


class SelectOrder(MatchRule):
    """
    A `MatchRule` that transforms the `order` into a set of integers as
    specified by a hyphenated string (e.g., 2-4). Used to parse orders for an
    `OrderedTransform`.

    Parameters
    ----------
    regex : str
        String representing the regular expression that indicates that a
        selected range of orders of the transform denoted `<begin>-<end>` are
        to be applied and concatenated. The regular expression must contain
        the symbols used to represent the transform as well as the exact
        parenthetic string `(?P<order>[0-9]+[\-]?[0-9]*)` in the position where
        the transform order is specified. Alongside the `select` argument, this
        is the regular expression that will be matched in any Expressions using
        this transform. Consult `PowerTransform` and `DerivativeTransform` for
        examples.
    """
    def __init__(self, regex):
        rule = lambda x: x.update(order=set(self._order_as_range(x['order'])))
        super(SelectOrder, self).__init__(regex=regex, rule=rule)

    def _order_as_range(self, order):
        """
        Convert a hyphenated string representing order of the column transform
        (if applicable) into a range object that can be passed as input to the
        appropriate expansion function. For instance, the string `3-5` will be
        converted to range(3, 6).
        """
        order = order.split('-')
        order = [int(o) for o in order]
        if len(order) > 1:
            order = range(order[0], (order[-1] + 1))
        return order
