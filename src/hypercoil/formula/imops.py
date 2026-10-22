# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Formula grammar for ``fslmaths``-like image operations.
"""
from __future__ import annotations
from dataclasses import field
from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Sequence,
    Tuple,
)

import jax
import jax.numpy as jnp
import nibabel as nb

from ..engine import Tensor
from .grammar import (
    Grammar,
    Grouping,
    GroupingPool,
    LeafInterpreter,
    Literalisation,
    TransformPool,
    TransformPrimitive,
)


def image_leaf_ingress(
    arg: Any,
) -> Callable:
    """Return the data array of a NIfTI image and its metadata."""
    return jnp.array(arg.get_fdata()), {
        'header': arg.header,
        'affine': arg.affine,
        'imtype': 'nifti',
    }


def scalar_leaf_ingress(
    arg: Any,
) -> Callable:
    """Return ``arg`` as a float, or else as a bool, with empty metadata."""
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
    """Return the first or the last ``num`` arguments of ``args``."""
    if side == 'left':
        return args[:num]
    else:
        return args[-num:]


def sub_regex(s: str) -> str:
    """Fill the number-type fields of ``s`` with their regular expressions."""
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
    """
    Form the regular expression of an ``fslmaths``-like command literal.

    The literal is ``-cmd``, then optional parameters in square brackets
    separated by ``|``, then an optional scalar in braces.

    Parameters
    ----------
    cmd
        Command name.
    params
        Type of each named parameter: ``'int'``, ``'float'``, ``'nnint'``,
        ``'nnfloat'`` or ``'nnintseq'`` (nonnegative integers separated by
        commas).
    unarised
        Type of the scalar operand of a unary form. None gives no scalar.
    right
        The scalar is followed by ``,...``: the scalar is the left operand.

    Returns
    -------
    str
        Regular expression.
    """
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
                param=param, kind=sub_regex(f'{{{typedict[kind]}}}')
            )
            for param, kind in params.items()
        )
        paramstr = r'\|?'.join(paramstr)
        paramstr = r'\[?' + paramstr + r'\]?'
    return f'[ ]?-{cmd}{paramstr}{scalar}[ ]?'


# TODO: This doesn't actually do anything yet. The last valid argument takes
#       all and the rest are ignored. There is no attempt to reconcile values.
#       This is a placeholder for future work.
def coalesce_metadata(*args: Tuple[Dict[str, Any], ...]) -> Dict[str, Any]:
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


# TODO: With this naive implementation, XLA supports morphological operations
#       only using rectangular structuring elements. This is a placeholder for
#       future work.
def morphology_rect_strel(
    image: Tensor,
    strel_shape: Tuple[int, ...],
    op: Literal['dilate', 'erode', 'open', 'close'],
    mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Perform a morphological operation using a rectangular structuring element.

    Parameters
    ----------
    image
        Image.
    strel_shape
        Shape of the structuring element, one size per axis of ``image``.
    op
        ``'dilate'``, ``'erode'``, ``'open'`` or ``'close'``.
    mask
        If given, each step changes only the voxels where ``mask`` is True.

    Returns
    -------
    Tensor
        Transformed image, of the shape of ``image``.
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


# TODO: Iterate until there are no further changes instead of requiring a fixed
#       number of iterations. We might lose differentiability, but I don't
#       think we can actually lose what we never had.
def morphology_rect_strel_fill_holes(
    image: Tensor,
    strel_shape: Tuple[int, ...],
    num_iters: int,
) -> Tensor:
    """
    Fill holes in a binary image using a rectangular structuring element.

    The background is grown inward from the border by masked dilation; voxels
    that it does not reach are holes.

    Parameters
    ----------
    image
        Binary image.
    strel_shape
        Shape of the structuring element, one size per axis of ``image``.
    num_iters
        Number of dilations. Holes are filled only where the background reaches
        within this number of steps.

    Returns
    -------
    Tensor
        Image with its holes filled.
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
    """
    Grammar of ``fslmaths``-like image operations.

    A compiled formula takes one argument per ``IMG`` leaf, in order, and
    returns the result array and the metadata of the images. Leaves that do
    not begin with ``IMG`` are scalars. Operations are written as suffixes,
    such as ``IMG -thr{0.5}`` or ``IMG -bin``, or as infixes between operands,
    such as ``IMG -add IMG``. Unary forms take a scalar in braces; the forms
    ``{s,...}`` of ``-sub``, ``-div`` and ``-mod`` put the scalar on the left.

    The operations are ``-bin``, ``-thr``, ``-uthr``, ``-dil``, ``-ero``,
    ``-opening``, ``-closing``, ``-fillholes``, ``-neg``, ``-or``, ``-and``,
    ``-add``, ``-sub``, ``-mul``, ``-div`` and ``-mod``.
    """
    groupings: GroupingPool = GroupingPool(
        Grouping(open='(', close=')'),
    )
    transforms: TransformPool = field(
        default_factory=lambda: TransformPool(
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
        )
    )
    whitespace: bool = True
    default_interpreter: Optional[LeafInterpreter] = field(
        default_factory=lambda: NiftiObjectInterpreter()
    )


class NiftiObjectInterpreter(LeafInterpreter):
    """Leaf interpreter that reads each ``IMG`` leaf from a NIfTI image."""
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
    """Leaf interpreter that loads each ``IMG`` leaf from a NIfTI file path."""
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


# ------------------------------- Binarise -------------------------------- #


class BinariseSuffixLiteralisation(Literalisation):
    """Literal ``-bin[t]`` of :class:`BinariseNode`; ``t`` defaults to 0."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='bin',
            params={
                'threshold': 'float',
            },
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the threshold; it defaults to 0."""
        threshold = params['threshold']
        if threshold == '':
            params['threshold'] = 0.0
        else:
            params['threshold'] = float(threshold)
        return params


class BinariseNode(TransformPrimitive):
    """Transform that binarises an image: True above the threshold."""
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (BinariseSuffixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled binarisation."""
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


# ------------------------------- Threshold ------------------------------- #


class UnaryThresholdSuffixLiteralisation(Literalisation):
    """Literal ``-thr[fill]{t}`` of :class:`UnaryThresholdNode`."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='thr',
            params={
                'fill': 'float',
            },
            unarised='float',
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the threshold and the fill value, which defaults to 0."""
        params['threshold'] = float(params['scalar'])
        fill_value = params['fill']
        if fill_value == '':
            params['fill'] = 0.0
        else:
            params['fill'] = float(fill_value)
        return params


class UnaryThresholdNode(TransformPrimitive):
    """Transform that sets voxels at or below a scalar to a fill value."""
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        UnaryThresholdSuffixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled threshold."""
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
    """Literal ``-thr[fill]`` of :class:`BinaryThresholdNode`."""
    affix: str = 'infix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='thr',
            params={
                'fill': 'float',
            },
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the fill value, which defaults to 0."""
        fill_value = params['fill']
        if fill_value == '':
            params['fill'] = 0.0
        else:
            params['fill'] = float(fill_value)
        return params


class BinaryThresholdNode(TransformPrimitive):
    """Transform that fills left voxels at or below the right operand."""
    min_arity: int = 2  # arg 0 is the model, arg 1 is the threshold value
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (
        BinaryThresholdInfixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled threshold."""
        f_lhs, f_rhs = pparams
        fill = params['fill']
        num_leaves = params['num_leaves']

        def threshold(lhs: Tensor, rhs: Tensor) -> Tensor:
            return jnp.where(lhs > rhs, lhs, fill)

        return binary_transform(threshold, f_lhs, f_rhs, num_leaves=num_leaves)


# ---------------------------- Upper Threshold ---------------------------- #


class UnaryUpperThresholdSuffixLiteralisation(Literalisation):
    """Literal ``-uthr[fill]{t}`` of :class:`UnaryUpperThresholdNode`."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='uthr',
            params={
                'fill': 'float',
            },
            unarised='float',
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the threshold and the fill value, which defaults to 0."""
        params['threshold'] = float(params['scalar'])
        fill_value = params['fill']
        if fill_value == '':
            params['fill'] = 0.0
        else:
            params['fill'] = float(fill_value)
        return params


class UnaryUpperThresholdNode(TransformPrimitive):
    """Transform that sets voxels at or above a scalar to a fill value."""
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        UnaryUpperThresholdSuffixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled upper threshold."""
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
    """Literal ``-uthr[fill]`` of :class:`UpperThresholdNode`."""
    affix: str = 'infix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='uthr',
            params={
                'fill': 'float',
            },
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the fill value, which defaults to 0."""
        fill_value = params['fill']
        if fill_value == '':
            params['fill'] = 0.0
        else:
            params['fill'] = float(fill_value)
        return params


class UpperThresholdNode(TransformPrimitive):
    """Transform that fills left voxels at or above the right operand."""
    min_arity: int = 2  # arg 0 is the model, arg 1 is the threshold value
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (UpperThresholdInfixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled upper threshold."""
        f_lhs, f_rhs = pparams
        fill = params['fill']
        num_leaves = params['num_leaves']

        def uthreshold(lhs: Tensor, rhs: Tensor) -> Tensor:
            return jnp.where(lhs < rhs, lhs, fill)

        return binary_transform(
            uthreshold, f_lhs, f_rhs, num_leaves=num_leaves
        )


# ------------------ Common to Morphological Transforms ------------------- #


class MorphologicalSuffixLiteralisation(Literalisation):
    """Base literal of the morphological transforms, with a kernel size."""
    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the kernel half-widths; they default to ``(1,)``."""
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
    return (
        morphology_rect_strel(
            img,
            kernel,
            op=op,
        ),
        meta,
    )


# ------------------------------- Dilation -------------------------------- #


class DilateSuffixLiteralisation(MorphologicalSuffixLiteralisation):
    """Literal ``-dil[k]`` of :class:`DilateNode`."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='dil',
            params={
                'kernel_size': 'nnintseq',
            },
        )
    )


class DilateNode(TransformPrimitive):
    """Transform that dilates an image with a box of half-width ``k``."""
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (DilateSuffixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled dilation."""
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


# -------------------------------- Erosion -------------------------------- #


class ErodeSuffixLiteralisation(MorphologicalSuffixLiteralisation):
    """Literal ``-ero[k]`` of :class:`ErodeNode`."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='ero',
            params={
                'kernel_size': 'nnintseq',
            },
        )
    )


class ErodeNode(TransformPrimitive):
    """Transform that erodes an image with a box of half-width ``k``."""
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (ErodeSuffixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled erosion."""
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


# -------------------------------- Opening -------------------------------- #


class OpeningSuffixLiteralisation(MorphologicalSuffixLiteralisation):
    """Literal ``-opening[k]`` of :class:`OpeningNode`."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='opening',
            params={
                'kernel_size': 'nnintseq',
            },
        )
    )


class OpeningNode(TransformPrimitive):
    """Transform that opens an image with a box of half-width ``k``."""
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (OpeningSuffixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled opening."""
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


# -------------------------------- Closing -------------------------------- #


class ClosingSuffixLiteralisation(MorphologicalSuffixLiteralisation):
    """Literal ``-closing[k]`` of :class:`ClosingNode`."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='closing',
            params={
                'kernel_size': 'nnintseq',
            },
        )
    )


class ClosingNode(TransformPrimitive):
    """Transform that closes an image with a box of half-width ``k``."""
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (ClosingSuffixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled closing."""
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


# ------------------------------ Fill Holes ------------------------------- #


class FillHolesSuffixLiteralisation(Literalisation):
    """Literal ``-fillholes[k|n]`` of :class:`FillHolesNode`."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='fillholes',
            params={
                'kernel_size': 'nnintseq',
                'num_iters': 'nnint',
            },
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the kernel half-widths and the number of iterations."""
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
    """Transform that fills the holes of a binary image."""
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (FillHolesSuffixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled hole filling."""
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
            return (
                morphology_rect_strel_fill_holes(
                    img,
                    kernel,
                    num_iters=n_iter,
                ),
                meta,
            )

        return closing


# ------------------------------- Negation -------------------------------- #


class NegationSuffixLiteralisation(Literalisation):
    """Literal ``-neg`` of :class:`NegationNode`."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='neg',
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``params`` unchanged."""
        return params


class NegationNode(TransformPrimitive):
    """Transform that gives the logical NOT of an image."""
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (NegationSuffixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled negation."""
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


# -------------------- General Binary Infix Operations -------------------- #


def curry_transform(*pparams, **params):
    """Return None."""
    return None


def binary_transform(
    transform: Callable,
    f_lhs: Callable,
    f_rhs: Callable,
    *,
    num_leaves: int,
) -> Callable:
    """
    Compile a binary operation on the outputs of two compiled operands.

    Parameters
    ----------
    transform
        Function of the left and the right operand.
    f_lhs
        Compiled left operand.
    f_rhs
        Compiled right operand.
    num_leaves
        Number of leaves of the operation, which selects its arguments.

    Returns
    -------
    Callable
        Compiled operation, returning the result and the coalesced metadata.
    """
    def transform_impl(
        *args,
        side: Literal['left', 'right'] = 'left',
    ) -> Tensor:
        args = select_args(args, side, num_leaves)
        img_lhs, meta_lhs = f_lhs(*args, side='left')
        img_rhs, meta_rhs = f_rhs(*args, side='right')
        return (
            transform(img_lhs, img_rhs),
            coalesce_metadata(meta_lhs, meta_rhs),
        )

    return transform_impl


# --------------------------------- Union --------------------------------- #


class UnionInfixLiteralisation(Literalisation):
    """Literal ``-or`` of :class:`UnionNode`."""
    affix: str = 'infix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='or',
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``params`` unchanged."""
        return params


class UnionNode(TransformPrimitive):
    """Transform that gives the logical OR of two operands."""
    min_arity: int = 2
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (UnionInfixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled union."""
        f_lhs, f_rhs = pparams
        num_leaves = params['num_leaves']

        def union(lhs: Tensor, rhs: Tensor) -> Tensor:
            return jnp.logical_or(lhs, rhs)

        return binary_transform(union, f_lhs, f_rhs, num_leaves=num_leaves)


# ----------------------------- Intersection ------------------------------ #


class IntersectionInfixLiteralisation(Literalisation):
    """Literal ``-and`` of :class:`IntersectionNode`."""
    affix: str = 'infix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='and',
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``params`` unchanged."""
        return params


class IntersectionNode(TransformPrimitive):
    """Transform that gives the logical AND of two operands."""
    min_arity: int = 2
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (IntersectionInfixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled intersection."""
        f_lhs, f_rhs = pparams
        num_leaves = params['num_leaves']

        def intersection(lhs: Tensor, rhs: Tensor) -> Tensor:
            return jnp.logical_and(lhs, rhs)

        return binary_transform(
            intersection,
            f_lhs,
            f_rhs,
            num_leaves=num_leaves,
        )


# ------------------------------- Addition -------------------------------- #


class UnaryAdditionSuffixLiteralisation(Literalisation):
    """Literal ``-add{s}`` of :class:`UnaryAdditionNode`."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='add',
            unarised='float',
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the scalar."""
        params['scalar'] = float(params['scalar'])
        return params


class UnaryAdditionNode(TransformPrimitive):
    """Transform that adds a scalar to an image."""
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (UnaryAdditionSuffixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled addition."""
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
    """Literal ``-add`` of :class:`AdditionNode`."""
    affix: str = 'infix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='add',
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``params`` unchanged."""
        return params


class AdditionNode(TransformPrimitive):
    """Transform that adds two operands."""
    min_arity: int = 2
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (AdditionInfixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled addition."""
        f_lhs, f_rhs = pparams
        num_leaves = params['num_leaves']

        def add(lhs: Tensor, rhs: Tensor) -> Tensor:
            return lhs + rhs

        return binary_transform(add, f_lhs, f_rhs, num_leaves=num_leaves)


# ------------------------------ Subtraction ------------------------------ #


class UnarySubtractionSuffixLeftLiteralisation(Literalisation):
    """Literal ``-sub{s}`` of :class:`UnarySubtractionNode`."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='sub',
            unarised='float',
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the scalar, the right operand."""
        params['scalar'] = float(params['scalar'])
        params['side'] = 'left'
        return params


class UnarySubtractionSuffixRightLiteralisation(Literalisation):
    """Literal ``-sub{s,...}`` of :class:`UnarySubtractionNode`."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='sub',
            unarised='float',
            right=True,
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the scalar, the left operand."""
        params['scalar'] = float(params['scalar'])
        params['side'] = 'right'
        return params


class UnarySubtractionNode(TransformPrimitive):
    """Transform that subtracts a scalar from an image, or an image from it."""
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        UnarySubtractionSuffixLeftLiteralisation(),
        UnarySubtractionSuffixRightLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled subtraction."""
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
    """Literal ``-sub`` of :class:`SubtractionNode`."""
    affix: str = 'infix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='sub',
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``params`` unchanged."""
        return params


class SubtractionNode(TransformPrimitive):
    """Transform that subtracts the right operand from the left."""
    min_arity: int = 2
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (SubtractionInfixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled subtraction."""
        f_lhs, f_rhs = pparams
        num_leaves = params['num_leaves']

        def subtract(lhs: Tensor, rhs: Tensor) -> Tensor:
            return lhs - rhs

        return binary_transform(subtract, f_lhs, f_rhs, num_leaves=num_leaves)


# ---------------------------- Multiplication ----------------------------- #


class UnaryMultiplicationSuffixLiteralisation(Literalisation):
    """Literal ``-mul{s}`` of :class:`UnaryMultiplicationNode`."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='mul',
            unarised='float',
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the scalar."""
        params['scalar'] = float(params['scalar'])
        return params


class UnaryMultiplicationNode(TransformPrimitive):
    """Transform that multiplies an image by a scalar."""
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        UnaryMultiplicationSuffixLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled multiplication."""
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
    """Literal ``-mul`` of :class:`MultiplicationNode`."""
    affix: str = 'infix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='mul',
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``params`` unchanged."""
        return params


class MultiplicationNode(TransformPrimitive):
    """Transform that multiplies two operands."""
    min_arity: int = 2
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (MultiplicationInfixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled multiplication."""
        f_lhs, f_rhs = pparams
        num_leaves = params['num_leaves']

        def multiply(lhs: Tensor, rhs: Tensor) -> Tensor:
            return lhs * rhs

        return binary_transform(multiply, f_lhs, f_rhs, num_leaves=num_leaves)


# ------------------------------- Division -------------------------------- #


class UnaryDivisionSuffixLeftLiteralisation(Literalisation):
    """Literal ``-div{s}`` of :class:`UnaryDivisionNode`."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='div',
            unarised='float',
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the scalar, the right operand."""
        params['scalar'] = float(params['scalar'])
        params['side'] = 'left'
        return params


class UnaryDivisionSuffixRightLiteralisation(Literalisation):
    """Literal ``-div{s,...}`` of :class:`UnaryDivisionNode`."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='div',
            unarised='float',
            right=True,
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the scalar, the left operand."""
        params['scalar'] = float(params['scalar'])
        params['side'] = 'right'
        return params


class UnaryDivisionNode(TransformPrimitive):
    """Transform that divides an image by a scalar, or a scalar by it."""
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        UnaryDivisionSuffixLeftLiteralisation(),
        UnaryDivisionSuffixRightLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled division."""
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
    """Literal ``-div`` of :class:`DivisionNode`."""
    affix: str = 'infix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='div',
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``params`` unchanged."""
        return params


class DivisionNode(TransformPrimitive):
    """
    Transform of the infix ``-div`` operation.

    The compiled operation computes the product of the operands, not their
    quotient.
    """
    min_arity: int = 2
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (DivisionInfixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled operation, which multiplies the operands."""
        f_lhs, f_rhs = pparams
        num_leaves = params['num_leaves']

        def divide(lhs: Tensor, rhs: Tensor) -> Tensor:
            return lhs * rhs

        return binary_transform(divide, f_lhs, f_rhs, num_leaves=num_leaves)


# ------------------------------- Remainder ------------------------------- #


class UnaryRemainderSuffixLeftLiteralisation(Literalisation):
    """Literal ``-mod{s}`` of :class:`UnaryRemainderNode`."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='mod',
            unarised='int',
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the scalar, the right operand."""
        params['scalar'] = float(params['scalar'])
        params['side'] = 'left'
        return params


class UnaryRemainderSuffixRightLiteralisation(Literalisation):
    """Literal ``-mod{s,...}`` of :class:`UnaryRemainderNode`."""
    affix: str = 'suffix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='mod',
            unarised='int',
            right=True,
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the scalar, the left operand."""
        params['scalar'] = float(params['scalar'])
        params['side'] = 'right'
        return params


class UnaryRemainderNode(TransformPrimitive):
    """Transform that gives an image modulo a scalar, or a scalar modulo it."""
    min_arity: int = 1
    max_arity: int = 1
    priority: int = 4
    literals: Sequence[Literalisation] = (
        UnaryRemainderSuffixLeftLiteralisation(),
        UnaryRemainderSuffixRightLiteralisation(),
    )

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled remainder."""
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
    """Literal ``-mod`` of :class:`RemainderNode`."""
    affix: str = 'infix'
    regex: str = field(
        default_factory=lambda: form_regex(
            cmd='mod',
        )
    )

    def parse_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Return ``params`` unchanged."""
        return params


class RemainderNode(TransformPrimitive):
    """Transform that gives the left operand modulo the right."""
    min_arity: int = 2
    max_arity: int = 2
    priority: int = 4
    literals: Sequence[Literalisation] = (RemainderInfixLiteralisation(),)

    def ascend(self, *pparams, **params) -> Callable:

        """Return the compiled remainder."""
        f_lhs, f_rhs = pparams
        num_leaves = params['num_leaves']

        def modulo(lhs: Tensor, rhs: Tensor) -> Tensor:
            return lhs % rhs

        return binary_transform(modulo, f_lhs, f_rhs, num_leaves=num_leaves)
