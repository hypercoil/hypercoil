# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Utilities
~~~~~~~~~
Utility functions for loading and manipulating data before they are ingested
by a model.
"""
import re
import numpy as np


def diff_nanpad(a, n=1, axis=-1):
    """
    Compute the nth-order difference of successive elements in an array and
    pad the beginning with NaN entries such that the resulting array is the
    same length as the input. Note that this differs from the `prepend`
    argument in `np.diff`.

    Parameters
    ----------
    a : ndarray
        Input array.
    n : int (default 1)
        Order of the difference, i.e., the number of times the difference
        is applied iteratively.
    axis : int (default -1)
        Axis along which the difference is computed.
    """
    diff = np.empty_like(a) * float('nan')
    diff[n:] = np.diff(a, n=n, axis=axis)
    return diff


def numbered_string(s):
    """
    Convert a string whose final characters are numeric into a tuple
    containing all characters except the final numeric ones in the first field
    and the numeric characters, cast to an int, in the second field. This
    permits comparison of strings of this form without regard to the number of
    zeros in the final numeric part. It also permits strings of this form to
    be sorted in numeric order.

    Parameters
    ----------
    s : string
        A string whose first characters are non-numeric and whose final
        characters are numeric.

    Returns
    -------
    tuple(str, int)
        Input string split into its initial non-numeric substring and its
        final numeric substring, cast to an int.
    """
    num = int(re.search('(?P<num>[0-9]+$)', s).groupdict()['num'])
    string = re.sub('[0-9]+$', '', s)
    return (string, num)


def match_metadata(pattern, metadata):
    """
    Find all dictionary keys that match a pattern.

    Parameters
    ----------
    pattern : compiled regular expression
        A compiled regular expression (obtained via re.compile) that
        represents the pattern to be matched in the metadata keys.
    metadata : dict
        Dictionary whose keys are to be searched for the specified pattern.
    """
    return list(filter(pattern.match, metadata.keys()))


def successive_pad_search(df, key, pad=0, k=5):
    """
    Find the column of a DataFrame that matches a query, ignoring potential
    zero-padding.

    Parameters
    ----------
    df : DataFrame
        DataFrame whose columns are to be searched for the specified key.
    key : str
        Key to search for in the DataFrame.
    pad : int (default 0)
        Number whose padding to ignore in the search
    k : int
        Maximum number of pads to try before a KeyError is raised.
    """
    for i in range(k):
        try:
            return df[key]
        except KeyError:
            p = '{:' + f'{pad}{i + 1}' + '}'
            st, nu = numbered_string(key)
            key = f'{st}' + p.format(nu)
    raise KeyError
