# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image maths
~~~~~~~~~~~
Formula grammar for ``fslmaths``-like operations.
"""
import jax.numpy as jnp
import nibabel as nb
from dataclasses import field
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple
from .grammar import (
    Grammar,
    Literalisation, TransformPrimitive,
    LeafInterpreter, Grouping, GroupingPool, TransformPool,
)
from ..engine import PyTree, Tensor


def image_leaf_ingress(
    arg: Any,
) -> Callable:
    return jnp.array(arg.get_fdata()), {
        'header': arg.header,
        'affine': arg.affine,
        'imtype': 'nifti',
    }


def scalar_leaf_ingress(
    arg: Any,
) -> Callable:
    try:
        out = float(arg)
    except ValueError:
        out = bool(arg)
    return out, {}


#TODO: This doesn't actually do anything yet. The last valid argument takes
#      all and the rest are ignored. There is no attempt to reconcile values.
#      This is a placeholder for future work.
def coalesce_metadata(
    *args: Tuple[Dict[str, Any], ...]
) -> Dict[str, Any]:
    """
    Coalesce metadata from a list of images.
    """
    out = {}
    for arg in args:
        header = arg.get('header', None)
        if header is not None:
            out['header'] = header
        affine = arg.get('affine', None)
        if affine is not None:
            out['affine'] = affine
    return out


class ImageMathsGrammar(Grammar):
    groupings: GroupingPool = GroupingPool(
        Grouping(open='(', close=')'),
    )
    transforms: TransformPool = field(
        default_factory = lambda: TransformPool(
            BinariseNode(),
            ThresholdNode(),
        ))
    whitespace: bool = True
    default_interpreter: Optional[LeafInterpreter] = field(
        default_factory = lambda: NiftiObjectInterpreter()
    )


class NiftiObjectInterpreter(LeafInterpreter):
    def __call__(self, leaf: Any) -> Callable:
        def img_and_meta(arg: Any) -> Tuple[Tensor, Dict[str, Any]]:
            if leaf == 'IMG':
                return image_leaf_ingress(arg)
            else:
                return scalar_leaf_ingress(leaf)
        return img_and_meta


class NiftiFileInterpreter(LeafInterpreter):
    def __call__(self, leaf: Any) -> Callable:
        def img_and_meta(arg: Any) -> Tuple[Tensor, Dict[str, Any]]:
            if leaf == 'IMG':
                obj = nb.load(arg)
                return image_leaf_ingress(obj)
            else:
                return scalar_leaf_ingress(leaf)
        return img_and_meta


#-------------------------------- Binarise ---------------------------------#


class BinariseSuffixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'infix'
    regex : str = r'[\ ]?-bin\[?(?P<threshold>[0-9]*\.*[0-9]*)\]?[\ ]?'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        threshold = params['threshold']
        if threshold == '':
            params['threshold'] = 0.0
        else:
            params['threshold'] = float(threshold)
        return params


class BinariseNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        BinariseSuffixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]
        threshold = params['threshold']

        def binarise(
            arg: Any,
        ) -> Tuple[Tensor, Any]:
            img, meta = f(arg)
            return jnp.where(img > threshold, 1.0, 0.0), meta

        return binarise


#-------------------------------- Threshold --------------------------------#


class ThresholdInfixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'infix'
    regex : str = r'[\ ]?-thr\[?(?P<fill>[0-9]*\.*[0-9]*)\]?[\ ]?'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        fill_value = params['fill']
        if fill_value == '':
            params['fill'] = 0.0
        else:
            params['fill'] = float(fill_value)
        return params


class ThresholdNode(TransformPrimitive):
    min_arity: int = 2 # arg 0 is the model, arg 1 is the threshold value
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (
        ThresholdInfixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f_acc, f_thr = pparams
        fill = params['fill']

        def threshold_curried(
            arg: Any,
        ) -> Tensor:
            img, meta_img = f_acc(arg)

            def _threshold_impl(
                arg: Any,
            ) -> Tensor:
                threshold, meta_thr = f_thr(arg)
                return (
                    jnp.where(img > threshold, img, fill),
                    coalesce_metadata(meta_thr, meta_img)
                )

            return _threshold_impl

        return threshold_curried


#----------------------------- Upper Threshold -----------------------------#


class UpperThresholdInfixLiteralisation(Literalisation):
    affix : Literal['prefix', 'suffix', 'infix', 'circumfix'] = 'infix'
    regex : str = r'[\ ]?-uthr\[?(?P<fill>[0-9]*\.*[0-9]*)\]?[\ ]?'

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        fill_value = params['fill']
        if fill_value == '':
            params['fill'] = 0.0
        else:
            params['fill'] = float(fill_value)
        return params


class UpperThresholdNode(TransformPrimitive):
    min_arity: int = 2 # arg 0 is the model, arg 1 is the threshold value
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (
        UpperThresholdInfixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f_acc, f_uthr = pparams
        fill = params['fill']

        def uthreshold_curried(
            arg: Any,
        ) -> Tensor:
            img, meta_img = f_acc(arg)

            def _uthreshold_impl(
                arg: Any,
            ) -> Tensor:
                threshold, meta_thr = f_uthr(arg)
                return (
                    jnp.where(img < threshold, img, fill),
                    coalesce_metadata(meta_thr, meta_img)
                )

            return _uthreshold_impl

        return uthreshold_curried


#-------------------------------- Dilation ---------------------------------#


#--------------------------------- Erosion ---------------------------------#


#--------------------------------- Opening ---------------------------------#


#--------------------------------- Closing ---------------------------------#


#------------------------------- Fill Holes --------------------------------#


#-------------------------------- Negation ---------------------------------#


#---------------------------------- Union ----------------------------------#


#------------------------------ Intersection -------------------------------#


#-------------------------------- Addition ---------------------------------#


#------------------------------- Subtraction -------------------------------#


#----------------------------- Multiplication ------------------------------#


#-------------------------------- Division ---------------------------------#


#-------------------------------- Remainder --------------------------------#
