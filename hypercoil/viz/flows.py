# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Visualisation control flow
~~~~~~~~~~~~~~~~~~~~~~~~~~
Functions for transforming the control flow of visualisation functions.
See also ``transforms.py`` for functions that transform the input and output
flows of visualisation functions.
"""
from itertools import chain
from typing import Literal, Mapping, Optional, Sequence
import numpy as np

from hypercoil.engine.axisutil import promote_axis


def source_chain(*pparams) -> callable:
    def transform(f: callable) -> callable:
        for p in reversed(pparams):
            f = p(f)
        return f
    return transform


def sink_chain(*pparams) -> callable:
    def transform(f: callable) -> callable:
        for p in pparams:
            f = p(f)
        return f
    return transform


def transform_chain(
    f: callable,
    source_chain: Optional[callable] = None,
    sink_chain: Optional[callable] = None,
) -> callable:
    if source_chain is not None:
        f = source_chain(f)
    if sink_chain is not None:
        f = sink_chain(f)
    return f


def split_chain(
    *chains: Sequence[callable],
) -> Sequence[callable]:
    def transform(f: callable) -> callable:
        fxfm = tuple(c(f) for c in chains)
        try:
            fxfm = tuple(chain(*fxfm))
        except TypeError:
            pass

        def f_transformed(**params: Mapping):
            return tuple(
                fx(**params)
                for fx in fxfm
            )

        return f_transformed
    return transform


def map_over_sequence(
    xfm: callable,
    mapping: Mapping[str, Sequence],
) -> callable:
    mapping_transform = close_mapping_transform(
        mapping,
    )
    def transform(f: callable) -> callable:
        return xfm(f, mapping_transform)
    return transform


def replicate_and_map(
    var: str,
    vals: Sequence,
) -> callable:
    def transform(f: callable) -> callable:
        def f_transformed(**params: Mapping):
            return tuple(
                f(**{**params, **{var: val}})
                for val in vals
            )

        return f_transformed
    return transform


def map_over_split_chain(
    *chains: Sequence[callable],
    mapping: Mapping[str, Sequence],
) -> callable:
    for vals in mapping.values():
        assert len(chains) == len(vals), "Number of chains must match number of values."
    def transform(f: callable) -> callable:
        fxfm = tuple(c(f) for c in chains)
        try:
            fxfm = tuple(chain(*fxfm))
        except TypeError:
            pass

        def f_transformed(**params: Mapping):
            return tuple(
                fxfm[i](**params, **{k: mapping[k][i] for k in mapping})
                for i in range(len(fxfm))
            )

        return f_transformed
    return transform


def apply_along_axis(var: str, axis: int) -> callable:
    def transform(f: callable) -> callable:
        def f_transformed(**params):
            val = params[var]
            new_ax = promote_axis(val.ndim, axis)
            val = np.transpose(val, new_ax)
            return tuple(
                f(**{**params, var: val[i]})
                for i in range(val.shape[0])
            )

        return f_transformed
    return transform


def direct_transform(
    f_outer: callable,
    f_inner: callable,
    unpack_dict: bool = False,
) -> callable:
    def transformed_f_outer(**f_outer_params):
        def transformed_f_inner(**f_inner_params):
            if unpack_dict:
                return f_outer(**{**f_outer_params, **f_inner(**f_inner_params)})
            return f_outer(f_inner(**f_inner_params), **f_outer_params)
        return transformed_f_inner
    return transformed_f_outer


def close_replicating_transform(
    mapping: Mapping,
    default_params: Literal["inner", "outer"] = "inner",
) -> callable:
    n_replicates = len(next(iter(mapping.values())))
    for v in mapping.values():
        assert len(v) == n_replicates, (
            "All mapped values must have the same length")
    def replicating_transform(
        f_outer: callable,
        f_inner: callable,
        unpack_dict: bool = False,
    ) -> callable:
        def transformed_f_outer(**f_outer_params):
            def transformed_f_inner(**f_inner_params):
                ret = []
                for i in range(n_replicates):
                    if default_params == "inner":
                        mapped_params_inner = {
                            k: v[i] for k, v in mapping.items()
                            if k not in f_outer_params
                        }
                        mapped_params_outer = {
                            k: v[i] for k, v in mapping.items()
                            if k in f_outer_params
                        }
                    elif default_params == "outer":
                        mapped_params_inner = {
                            k: v[i] for k, v in mapping.items()
                            if k in f_inner_params
                        }
                        mapped_params_outer = {
                            k: v[i] for k, v in mapping.items()
                            if k not in f_inner_params
                        }
                    f_inner_params_i = {
                        **f_inner_params,
                        **mapped_params_inner
                    }
                    f_outer_params_i = {
                        **f_outer_params,
                        **mapped_params_outer
                    }
                    if unpack_dict:
                        ret.append(
                            f_outer(**{**f_inner(**f_inner_params_i), **f_outer_params_i})
                        )
                    else:
                        ret.append(
                            f_outer(f_inner(**f_inner_params_i)[i], **f_outer_params_i)
                        )
                return tuple(ret)
            return transformed_f_inner
        return transformed_f_outer
    return replicating_transform


def close_mapping_transform(
    mapping: Mapping,
) -> callable:
    n_replicates = len(next(iter(mapping.values())))
    for v in mapping.values():
        assert len(v) == n_replicates, (
            "All mapped values must have the same length")
    def mapping_transform(
        f_outer: callable,
        f_inner: callable,
        unpack_dict: bool = False,
    ) -> callable:
        def transformed_f_outer(**f_outer_params):
            def transformed_f_inner(**f_inner_params):
                ret = []
                out = f_inner(**f_inner_params)
                assert len(out) == n_replicates, (
                    "The length of the output of the inner function must be "
                    "equal to the length of the mapped values")
                for i, o in enumerate(out):
                    mapped_params = {k: v[i] for k, v in mapping.items()}
                    if unpack_dict:
                        f_outer_params_i = {
                            **f_outer_params,
                            **mapped_params,
                            **o
                        }
                        ret.append(f_outer(**f_outer_params_i))
                    else:
                        f_outer_params_i = {
                            **f_outer_params,
                            **mapped_params,
                        }
                        ret.append(f_outer(o, **f_outer_params_i))
                return tuple(ret)
            return transformed_f_inner
        return transformed_f_outer
    return mapping_transform
