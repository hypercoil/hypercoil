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


def _seq_to_dict(
    seq: Sequence[Mapping],
    chain_vars: Sequence[str] = ("plotter",),
    shared_keys: bool = True,
) -> Mapping[str, Sequence]:
    if shared_keys:
        keys = seq[0].keys()
    else:
        keys = [set(s) for s in seq.keys()]
        keys = set.intersection(*keys)
    dct = {k: tuple(r[k] for r in seq) for k in keys}
    for k in dct:
        try:
            dct[k] = tuple(chain(*dct[k]))
        except TypeError:
            pass
    # for k in chain_vars:
    #     if k not in dct:
    #         continue
    #     try:
    #         dct[k] = tuple(chain(*dct[k]))
    #     except TypeError:
    #         pass
    return dct


def _dict_to_seq(
    dct: Mapping[str, Sequence],
) -> Sequence[Mapping]:
    keys = dct.keys()
    seq = tuple(
        dict(zip(dct.keys(), v))
        for v in zip(*dct.values())
    )
    return seq


def ichain(*pparams) -> callable:
    def transform(f: callable) -> callable:
        for p in reversed(pparams):
            f = p(f)
        return f
    return transform


def ochain(*pparams) -> callable:
    def transform(f: callable) -> callable:
        for p in pparams:
            f = p(f)
        return f
    return transform


def iochain(
    f: callable,
    ichain: Optional[callable] = None,
    ochain: Optional[callable] = None,
) -> callable:
    if ichain is not None:
        f = ichain(f)
    if ochain is not None:
        f = ochain(f)
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
            ret = tuple(
                fx(**params)
                for fx in fxfm
            )
            return _seq_to_dict(ret)

        return f_transformed
    return transform


def map_over_sequence(
    xfm: callable,
    mapping: Optional[Mapping[str, Sequence]] = None,
    n_replicates: Optional[int] = None,
) -> callable:
    mapping_transform = close_mapping_transform(
        mapping=mapping,
        n_replicates=n_replicates,
    )
    def transform(f: callable) -> callable:
        return xfm(f, mapping_transform)
    return transform


def replicate_and_map(
    xfm: callable,
    mapping: Mapping[str, Sequence],
    default_params: Literal["inner", "outer"] = "inner",
) -> callable:
    mapping_transform = close_replicating_transform(
        mapping,
        default_params=default_params,
    )
    def transform(f: callable) -> callable:
        return xfm(f, mapping_transform)
    return transform


def replicate(
    mapping: Mapping[str, Sequence],
) -> callable:
    n_vals = len(next(iter(mapping.values())))
    for vals in mapping.values():
        assert len(vals) == n_vals, (
            "All values must have the same length. Perhaps you intended to "
            "nest replications?"
        )
    def transform(f: callable) -> callable:
        def f_transformed(**params: Mapping):
            ret = []
            for i in range(n_vals):
                nxt = f(**{**params, **{k: mapping[k][i] for k in mapping}, **{"copy_actors": True}})
                ret += [nxt]
            return _seq_to_dict(ret)

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
            ret = tuple(
                f(**{**params, var: val[i]})
                for i in range(val.shape[0])
            )
            return _seq_to_dict(ret)

        return f_transformed
    return transform


def direct_transform(
    f_outer: callable,
    f_inner: callable,
    unpack_dict: bool = True,
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
                return _seq_to_dict(ret)
            return transformed_f_inner
        return transformed_f_outer
    return replicating_transform


def close_mapping_transform(
    mapping: Optional[Mapping] = None,
    n_replicates: Optional[int] = None,
) -> callable:
    if n_replicates is None:
        n_replicates = len(next(iter(mapping.values())))
    if mapping is None:
        mapping = {}
    for v in mapping.values():
        assert len(v) == n_replicates, (
            "All mapped values must have the same length")
    def mapping_transform(
        f_outer: callable,
        f_inner: callable,
        unpack_dict: bool = True,
    ) -> callable:
        def transformed_f_outer(**f_outer_params):
            def transformed_f_inner(**f_inner_params):
                ret = []
                out = f_inner(**f_inner_params)
                out = _dict_to_seq(out)
                assert len(out) == n_replicates, (
                    "The length of the output of the inner function must be "
                    "equal to the length of the mapped values")
                for i, o in enumerate(out):
                    if mapping:
                        mapped_params = {k: v[i] for k, v in mapping.items()}
                    if unpack_dict:
                        f_outer_params_i = {
                            **f_outer_params,
                            **mapped_params,
                            **o,
                        }
                        ret.append(f_outer(**f_outer_params_i))
                    else:
                        f_outer_params_i = {
                            **f_outer_params,
                            **mapped_params,
                        }
                        ret.append(f_outer(o, **f_outer_params_i))
                return _seq_to_dict(ret)
            return transformed_f_inner
        return transformed_f_outer
    return mapping_transform
