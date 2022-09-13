# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Image maths
~~~~~~~~~~~
Formula grammar for ``fslmaths``-like operations.
"""
import jax
import jax.numpy as jnp
import nibabel as nb
from dataclasses import field
from typing import Any, Callable, Dict, Literal, Optional, Sequence, Tuple
from .grammar import (
    Grammar,
    Literalisation, TransformPrimitive,
    LeafInterpreter, Grouping, GroupingPool, TransformPool,
)
from ..engine import Tensor


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


def select_args(
    args: Sequence,
    side: Literal['left', 'right'],
    num: int,
):
    if side == 'left':
        return args[:num]
    else:
        return args[-num:]


def sub_regex(s: str) -> str:
    i = r'\-?[0-9]*'
    f = r'\-?[0-9]*\.*[0-9]*'
    nni = r'[0-9]*'
    nnf = r'[0-9]*\.*[0-9]*'
    nnii = r'[0-9]*[0-9\,]*'
    return s.format(
        i=i,
        f=f,
        nni=nni,
        nnf=nnf,
        nnii=nnii,
    )


def form_regex(
    cmd: str,
    params: Optional[Dict[str, str]] = None,
    unarised: Optional[str] = None,
    right: bool = False,
):
    typedict = {
        'int': 'i',
        'float': 'f',
        'nnint': 'nni',
        'nnfloat': 'nnf',
        'nnintseq': 'nnii',
    }
    scalar = ''
    paramstr = ''
    right = r'\,\.\.\.' if right else ''
    if unarised is not None:
        unarised = typedict[unarised]
        scalar = '{{(?P<scalar>{}){}}}'.format(
            sub_regex(f'{{{unarised}}}'),
            right,
        )
    if params is not None:
        paramstr = tuple(
            '(?P<{param}>{kind})'.format(
                param=param,
                kind=sub_regex(f'{{{typedict[kind]}}}')
            ) for param, kind in params.items()
        )
        paramstr = r'\|?'.join(paramstr)
        paramstr = r'\[?' + paramstr + r'\]?'
    return f'[ ]?-{cmd}{paramstr}{scalar}[ ]?'


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


#TODO: With this naive implementation, XLA supports morphological operations
#      only using rectangular structuring elements. This is a placeholder for
#      future work.
def morphology_rect_strel(
    image: Tensor,
    strel_shape: Tuple[int, ...],
    op: Literal['dilate', 'erode', 'open', 'close'],
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Perform a morphological operation using a rectangular structuring element.
    """

    op_series = {
        'dilate': ('dilate',),
        'erode': ('erode',),
        'open': ('erode', 'dilate'),
        'close': ('dilate', 'erode'),
    }
    op_dict = {
        'dilate': (-float('inf'), jax.lax.max),
        'erode': (float('inf'), jax.lax.min),
    }

    stride = tuple(1 for _ in strel_shape)
    padding = tuple((s // 2, s // 2) for s in strel_shape)
    def transform(data, init, computation):
        return jax.lax.reduce_window(
            data,
            init,
            computation=computation,
            window_dimensions=(1,) + strel_shape,
            window_strides=(1,) + stride,
            padding=((0, 0),) + padding,
        )

    data = image[None, ...]
    for op in op_series[op]:
        init, computation = op_dict[op]
        data = transform(data, init, computation)
        if mask is not None:
            mask = mask[None, ...]
            data = jnp.where(mask, data, image[None, ...])
    return data[0]


#TODO: Iterate until there are no further changes instead of requiring a fixed
#      number of iterations. We might lose differentiability, but I don't
#      think we can actually lose what we never had.
def morphology_rect_strel_fill_holes(
    image: Tensor,
    strel_shape: Tuple[int, ...],
    num_iters: int,
) -> Tensor:
    """
    Fill holes in a binary image using a rectangular structuring element.
    """
    padding = ((1, 1),) * image.ndim
    mask = jnp.logical_not(image)
    data = jnp.zeros_like(mask).astype(bool)
    data = jnp.pad(data, padding, constant_values=True)
    mask = jnp.pad(mask, padding, constant_values=False)
    for _ in range(num_iters):
        data = morphology_rect_strel(data, strel_shape, 'dilate', mask)
    slices = tuple(slice(1, -1) for _ in range(image.ndim))
    return jnp.logical_not(data)[slices]


class ImageMathsGrammar(Grammar):
    groupings: GroupingPool = GroupingPool(
        Grouping(open='(', close=')'),
    )
    transforms: TransformPool = field(
        default_factory = lambda: TransformPool(
            BinariseNode(),
            UnaryThresholdNode(),
            BinaryThresholdNode(),
            UnaryUpperThresholdNode(),
            UpperThresholdNode(),
            DilateNode(),
            ErodeNode(),
            OpeningNode(),
            ClosingNode(),
            FillHolesNode(),
            NegationNode(),
            UnionNode(),
            IntersectionNode(),
            UnaryAdditionNode(),
            AdditionNode(),
            UnarySubtractionNode(),
            SubtractionNode(),
            UnaryMultiplicationNode(),
            MultiplicationNode(),
            UnaryDivisionNode(),
            DivisionNode(),
            UnaryRemainderNode(),
            RemainderNode(),
        ))
    whitespace: bool = True
    default_interpreter: Optional[LeafInterpreter] = field(
        default_factory = lambda: NiftiObjectInterpreter()
    )


class NiftiObjectInterpreter(LeafInterpreter):
    def __call__(self, leaf: Any) -> Callable:
        def img_and_meta(
            *args: Any,
            side: Literal['left', 'right'] = 'left',
        ) -> Tuple[Tensor, Dict[str, Any]]:
            arg = select_args(args, side=side, num=1)[0]
            if leaf[:3] == 'IMG':
                return image_leaf_ingress(arg)
            else:
                return scalar_leaf_ingress(leaf)
        return img_and_meta


class NiftiFileInterpreter(LeafInterpreter):
    def __call__(self, leaf: Any) -> Callable:
        def img_and_meta(
            *args: Any,
            side: Literal['left', 'right'] = 'left',
        ) -> Tuple[Tensor, Dict[str, Any]]:
            arg = select_args(args, side=side, num=1)[0]
            if leaf[:3] == 'IMG':
                obj = nb.load(arg)
                return image_leaf_ingress(obj)
            else:
                return scalar_leaf_ingress(leaf)
        return img_and_meta


#-------------------------------- Binarise ---------------------------------#


class BinariseSuffixLiteralisation(Literalisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='bin',
        params={
            'threshold': 'float',
        },
    ))

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
        num_leaves = params['num_leaves']

        def binarise(
            *args,
            side: Literal['left', 'right'] = 'left',
        ) -> Tuple[Tensor, Any]:
            args = select_args(args, side, num_leaves)
            img, meta = f(*args, side='left')
            return jnp.where(img > threshold, True, False), meta

        return binarise


#-------------------------------- Threshold --------------------------------#


class UnaryThresholdSuffixLiteralisation(Literalisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='thr',
        params={
            'fill': 'float',
        },
        unarised='float',
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['threshold'] = float(params['scalar'])
        fill_value = params['fill']
        if fill_value == '':
            params['fill'] = 0.0
        else:
            params['fill'] = float(fill_value)
        return params


class UnaryThresholdNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        UnaryThresholdSuffixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]
        thresh = params['threshold']
        fill_value = params['fill']
        num_leaves = params['num_leaves']

        def threshold(
            *args,
            side: Literal['left', 'right'] = 'left',
        ) -> Tuple[Tensor, Any]:
            args = select_args(args, side, num_leaves)
            img, meta = f(*args, side='left')
            return jnp.where(img > thresh, img, fill_value), meta

        return threshold


class BinaryThresholdInfixLiteralisation(Literalisation):
    affix : str = 'infix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='thr',
        params={
            'fill': 'float',
        },
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        fill_value = params['fill']
        if fill_value == '':
            params['fill'] = 0.0
        else:
            params['fill'] = float(fill_value)
        return params


class BinaryThresholdNode(TransformPrimitive):
    min_arity: int = 2 # arg 0 is the model, arg 1 is the threshold value
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (
        BinaryThresholdInfixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f_lhs, f_rhs = pparams
        fill = params['fill']
        num_leaves = params['num_leaves']

        def threshold(lhs: Tensor, rhs: Tensor) -> Tensor:
            return jnp.where(lhs > rhs, lhs, fill)

        return binary_transform(threshold, f_lhs, f_rhs, num_leaves=num_leaves)


#----------------------------- Upper Threshold -----------------------------#


class UnaryUpperThresholdSuffixLiteralisation(Literalisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='uthr',
        params={
            'fill': 'float',
        },
        unarised='float',
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['threshold'] = float(params['scalar'])
        fill_value = params['fill']
        if fill_value == '':
            params['fill'] = 0.0
        else:
            params['fill'] = float(fill_value)
        return params


class UnaryUpperThresholdNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        UnaryUpperThresholdSuffixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]
        thresh = params['threshold']
        fill_value = params['fill']
        num_leaves = params['num_leaves']

        def threshold(
            *args,
            side: Literal['left', 'right'] = 'left',
        ) -> Tuple[Tensor, Any]:
            args = select_args(args, side, num_leaves)
            img, meta = f(*args, side='left')
            return jnp.where(img < thresh, img, fill_value), meta

        return threshold


class UpperThresholdInfixLiteralisation(Literalisation):
    affix : str = 'infix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='uthr',
        params={
            'fill': 'float',
        },
    ))

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

        f_lhs, f_rhs = pparams
        fill = params['fill']
        num_leaves = params['num_leaves']

        def uthreshold(lhs: Tensor, rhs: Tensor) -> Tensor:
            return jnp.where(lhs < rhs, lhs, fill)

        return binary_transform(uthreshold, f_lhs, f_rhs, num_leaves=num_leaves)


#------------------- Common to Morphological Transforms --------------------#


class MorphologicalSuffixLiteralisation(Literalisation):
    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        kernel_size = params['kernel_size']
        if kernel_size == '':
            params['kernel_size'] = (1,)
        else:
            kernel_size = kernel_size.split(',')
            kernel_size = tuple(int(k) for k in kernel_size)
            params['kernel_size'] = kernel_size
        return params


def _morphological_transform(
    img: Tensor,
    meta: Dict[str, Any],
    kernel_size: Tuple[int, ...],
    op: str,
) -> Tuple[Tensor, Dict[str, Any]]:

    if len(kernel_size) == 1:
        kernel = kernel_size * img.ndim
    else:
        kernel = kernel_size
    kernel = tuple(k * 2 + 1 for k in kernel)
    return morphology_rect_strel(
        img,
        kernel,
        op=op,
    ), meta


#-------------------------------- Dilation ---------------------------------#


class DilateSuffixLiteralisation(MorphologicalSuffixLiteralisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='dil',
        params={
            'kernel_size': 'nnintseq',
        },
    ))


class DilateNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        DilateSuffixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]
        kernel_size = params['kernel_size']
        num_leaves = params['num_leaves']

        def dilate(
            *args,
            side: Literal['left', 'right'] = 'left',
        ) -> Tensor:
            args = select_args(args, side, num_leaves)
            img, meta = f(*args)
            return _morphological_transform(
                img=img,
                meta=meta,
                kernel_size=kernel_size,
                op='dilate',
            )

        return dilate


#--------------------------------- Erosion ---------------------------------#


class ErodeSuffixLiteralisation(MorphologicalSuffixLiteralisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='ero',
        params={
            'kernel_size': 'nnintseq',
        },
    ))


class ErodeNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        ErodeSuffixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]
        kernel_size = params['kernel_size']
        num_leaves = params['num_leaves']

        def erode(
            *args,
            side: Literal['left', 'right'] = 'left',
        ) -> Tensor:
            args = select_args(args, side, num_leaves)
            img, meta = f(*args)
            return _morphological_transform(
                img=img,
                meta=meta,
                kernel_size=kernel_size,
                op='erode',
            )

        return erode


#--------------------------------- Opening ---------------------------------#


class OpeningSuffixLiteralisation(MorphologicalSuffixLiteralisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='opening',
        params={
            'kernel_size': 'nnintseq',
        },
    ))


class OpeningNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        OpeningSuffixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]
        kernel_size = params['kernel_size']
        num_leaves = params['num_leaves']

        def opening(
            *args,
            side: Literal['left', 'right'] = 'left',
        ) -> Tensor:
            args = select_args(args, side, num_leaves)
            img, meta = f(*args)
            return _morphological_transform(
                img=img,
                meta=meta,
                kernel_size=kernel_size,
                op='open',
            )

        return opening


#--------------------------------- Closing ---------------------------------#


class ClosingSuffixLiteralisation(MorphologicalSuffixLiteralisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='closing',
        params={
            'kernel_size': 'nnintseq',
        },
    ))


class ClosingNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        ClosingSuffixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]
        kernel_size = params['kernel_size']
        num_leaves = params['num_leaves']

        def closing(
            *args,
            side: Literal['left', 'right'] = 'left',
        ) -> Tensor:
            args = select_args(args, side, num_leaves)
            img, meta = f(*args)
            return _morphological_transform(
                img=img,
                meta=meta,
                kernel_size=kernel_size,
                op='close',
            )

        return closing


#------------------------------- Fill Holes --------------------------------#


class FillHolesSuffixLiteralisation(Literalisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='fillholes',
        params={
            'kernel_size': 'nnintseq',
            'num_iters': 'nnint',
        },
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        num_iters = params.get('num_iters', None)
        if num_iters is not None:
            num_iters = int(num_iters)
        params['num_iters'] = num_iters

        kernel_size = params['kernel_size']
        if kernel_size == '':
            params['kernel_size'] = (1,)
        else:
            kernel_size = kernel_size.split(',')
            kernel_size = tuple(int(k) for k in kernel_size)
            params['kernel_size'] = kernel_size
        return params


class FillHolesNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        FillHolesSuffixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]
        kernel_size = params['kernel_size']
        num_iters = params['num_iters']
        num_leaves = params['num_leaves']

        def closing(
            *args,
            side: Literal['left', 'right'] = 'left',
        ) -> Tensor:
            args = select_args(args, side, num_leaves)
            img, meta = f(*args)
            if len(kernel_size) == 1:
                kernel = kernel_size * img.ndim
            else:
                kernel = kernel_size
            if num_iters is None:
                n_iter = max(img.shape) // 2
            else:
                n_iter = num_iters
            kernel = tuple(k * 2 + 1 for k in kernel)
            return morphology_rect_strel_fill_holes(
                img,
                kernel,
                num_iters=n_iter,
            ), meta

        return closing


#-------------------------------- Negation ---------------------------------#


class NegationSuffixLiteralisation(Literalisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='neg',
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class NegationNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        NegationSuffixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]
        num_leaves = params['num_leaves']

        def negate(
            *args,
            side: Literal['left', 'right'] = 'left',
        ) -> Tensor:
            args = select_args(args, side, num_leaves)
            img, meta = f(*args)
            return jnp.logical_not(img), meta

        return negate


#--------------------- General Binary Infix Operations ---------------------#


def curry_transform(*pparams, **params): return None


def binary_transform(
    transform: Callable,
    f_lhs: Callable,
    f_rhs: Callable,
    *,
    num_leaves: int,
) -> Callable:

    def transform_impl(
        *args,
        side: Literal['left', 'right'] = 'left',
    ) -> Tensor:
        args = select_args(args, side, num_leaves)
        img_lhs, meta_lhs = f_lhs(*args, side='left')
        img_rhs, meta_rhs = f_rhs(*args, side='right')
        return (
            transform(img_lhs, img_rhs),
            coalesce_metadata(meta_lhs, meta_rhs)
        )

    return transform_impl


#---------------------------------- Union ----------------------------------#


class UnionInfixLiteralisation(Literalisation):
    affix : str = 'infix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='or',
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class UnionNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (
        UnionInfixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f_lhs, f_rhs = pparams
        num_leaves = params['num_leaves']

        def union(lhs: Tensor, rhs: Tensor) -> Tensor:
            return jnp.logical_or(lhs, rhs)

        return binary_transform(union, f_lhs, f_rhs, num_leaves=num_leaves)


#------------------------------ Intersection -------------------------------#


class IntersectionInfixLiteralisation(Literalisation):
    affix : str = 'infix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='and',
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class IntersectionNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (
        IntersectionInfixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f_lhs, f_rhs = pparams
        num_leaves = params['num_leaves']

        def intersection(lhs: Tensor, rhs: Tensor) -> Tensor:
            return jnp.logical_and(lhs, rhs)

        return binary_transform(intersection, f_lhs, f_rhs, num_leaves=num_leaves)


#-------------------------------- Addition ---------------------------------#


class UnaryAdditionSuffixLiteralisation(Literalisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='add',
        unarised='float',
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['scalar'] = float(params['scalar'])
        return params


class UnaryAdditionNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        UnaryAdditionSuffixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]
        scalar = params['scalar']
        num_leaves = params['num_leaves']

        def add(
            *args,
            side: Literal['left', 'right'] = 'left',
        ) -> Tuple[Tensor, Any]:
            args = select_args(args, side, num_leaves)
            img, meta = f(*args, side='left')
            return img + scalar, meta

        return add


class AdditionInfixLiteralisation(Literalisation):
    affix : str = 'infix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='add',
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class AdditionNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (
        AdditionInfixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f_lhs, f_rhs = pparams
        num_leaves = params['num_leaves']

        def add(lhs: Tensor, rhs: Tensor) -> Tensor:
            return lhs + rhs

        return binary_transform(add, f_lhs, f_rhs, num_leaves=num_leaves)


#------------------------------- Subtraction -------------------------------#


class UnarySubtractionSuffixLeftLiteralisation(Literalisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='sub',
        unarised='float',
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['scalar'] = float(params['scalar'])
        params['side'] = 'left'
        return params


class UnarySubtractionSuffixRightLiteralisation(Literalisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='sub',
        unarised='float',
        right=True,
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['scalar'] = float(params['scalar'])
        params['side'] = 'right'
        return params


class UnarySubtractionNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        UnarySubtractionSuffixLeftLiteralisation(),
        UnarySubtractionSuffixRightLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]
        scalar = params['scalar']
        op_side = params['side']
        num_leaves = params['num_leaves']

        def subtract(
            *args,
            side: Literal['left', 'right'] = 'left',
        ) -> Tuple[Tensor, Any]:
            args = select_args(args, side, num_leaves)
            img, meta = f(*args, side='left')
            if op_side == 'left':
                return img - scalar, meta
            else:
                return scalar - img, meta

        return subtract


class SubtractionInfixLiteralisation(Literalisation):
    affix : str = 'infix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='sub',
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class SubtractionNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (
        SubtractionInfixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f_lhs, f_rhs = pparams
        num_leaves = params['num_leaves']

        def subtract(lhs: Tensor, rhs: Tensor) -> Tensor:
            return lhs - rhs

        return binary_transform(subtract, f_lhs, f_rhs, num_leaves=num_leaves)


#----------------------------- Multiplication ------------------------------#


class UnaryMultiplicationSuffixLiteralisation(Literalisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='mul',
        unarised='float',
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['scalar'] = float(params['scalar'])
        return params


class UnaryMultiplicationNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        UnaryMultiplicationSuffixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]
        scalar = params['scalar']
        num_leaves = params['num_leaves']

        def multiply(
            *args,
            side: Literal['left', 'right'] = 'left',
        ) -> Tuple[Tensor, Any]:
            args = select_args(args, side, num_leaves)
            img, meta = f(*args, side='left')
            return img * scalar, meta

        return multiply


class MultiplicationInfixLiteralisation(Literalisation):
    affix : str = 'infix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='mul',
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class MultiplicationNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (
        MultiplicationInfixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f_lhs, f_rhs = pparams
        num_leaves = params['num_leaves']

        def multiply(lhs: Tensor, rhs: Tensor) -> Tensor:
            return lhs * rhs

        return binary_transform(multiply, f_lhs, f_rhs, num_leaves=num_leaves)


#-------------------------------- Division ---------------------------------#


class UnaryDivisionSuffixLeftLiteralisation(Literalisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='div',
        unarised='float',
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['scalar'] = float(params['scalar'])
        params['side'] = 'left'
        return params


class UnaryDivisionSuffixRightLiteralisation(Literalisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='div',
        unarised='float',
        right=True,
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['scalar'] = float(params['scalar'])
        params['side'] = 'right'
        return params


class UnaryDivisionNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        UnaryDivisionSuffixLeftLiteralisation(),
        UnaryDivisionSuffixRightLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]
        scalar = params['scalar']
        op_side = params['side']
        num_leaves = params['num_leaves']

        def divide(
            *args,
            side: Literal['left', 'right'] = 'left',
        ) -> Tuple[Tensor, Any]:
            args = select_args(args, side, num_leaves)
            img, meta = f(*args, side='left')
            if op_side == 'left':
                return img / scalar, meta
            else:
                return scalar / img, meta

        return divide


class DivisionInfixLiteralisation(Literalisation):
    affix : str = 'infix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='div',
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class DivisionNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (
        DivisionInfixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f_lhs, f_rhs = pparams
        num_leaves = params['num_leaves']

        def divide(lhs: Tensor, rhs: Tensor) -> Tensor:
            return lhs * rhs

        return binary_transform(divide, f_lhs, f_rhs, num_leaves=num_leaves)


#-------------------------------- Remainder --------------------------------#


class UnaryRemainderSuffixLeftLiteralisation(Literalisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='mod',
        unarised='int',
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['scalar'] = float(params['scalar'])
        params['side'] = 'left'
        return params


class UnaryRemainderSuffixRightLiteralisation(Literalisation):
    affix : str = 'suffix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='mod',
        unarised='int',
        right=True,
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        params['scalar'] = float(params['scalar'])
        params['side'] = 'right'
        return params


class UnaryRemainderNode(TransformPrimitive):
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        UnaryRemainderSuffixLeftLiteralisation(),
        UnaryRemainderSuffixRightLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f = pparams[0]
        scalar = params['scalar']
        op_side = params['side']
        num_leaves = params['num_leaves']

        def modulo(
            *args,
            side: Literal['left', 'right'] = 'left',
        ) -> Tuple[Tensor, Any]:
            args = select_args(args, side, num_leaves)
            img, meta = f(*args, side='left')
            if op_side == 'left':
                return img % scalar, meta
            else:
                return scalar % img, meta

        return modulo


class RemainderInfixLiteralisation(Literalisation):
    affix : str = 'infix'
    regex : str = field(default_factory = lambda: form_regex(
        cmd='mod',
    ))

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params


class RemainderNode(TransformPrimitive):
    min_arity: int = 2
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (
        RemainderInfixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        f_lhs, f_rhs = pparams
        num_leaves = params['num_leaves']

        def modulo(lhs: Tensor, rhs: Tensor) -> Tensor:
            return lhs % rhs

        return binary_transform(modulo, f_lhs, f_rhs, num_leaves=num_leaves)
