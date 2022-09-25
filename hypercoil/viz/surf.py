# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Brain surfaces
~~~~~~~~~~~~~~
Brain surface objects for plotting.
"""
import pathlib
import pyvista as pv
import nibabel as nb
import templateflow.api as tflow

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple, Union
from hypercoil.engine import Tensor
from hypercoil.neuro.const import (
    template_dict
)


def is_path_like(obj: Any) -> bool:
    return isinstance(obj, (str, pathlib.Path))


# class ProjectionKey:
#     """
#     Currently, PyVista does not support non-string keys for point data. This class
#     is unused, but is a placeholder in case PyVista supports this in the future.
#     """
#     def __init__(self, projection: str):
#         self.projection = projection

#     def __hash__(self):
#         return hash(f'_projection_{self.projection}')

#     def __eq__(self, other):
#         return self.projection == other.projection

#     def __repr__(self):
#         return f'Projection({self.projection})'


@dataclass
class ProjectedPolyData(pv.PolyData):
    """
    PyVista ``PolyData`` object with multiple projections.
    """

    def __init__(
        self,
        *pparams,
        projection: str,
        **params
    ):
        super().__init__(*pparams, **params)
        self.point_data[f'_projection_{projection}'] = self.points

    def __repr__(self):
        return super().__repr__()

    @property
    def projections(self):
        return {
            k[12:]: v for k, v in self.point_data.items()
            if k[:12] == '_projection_'
        }

    @classmethod
    def from_polydata(
        cls,
        *,
        projection: str,
        ignore_errors: bool = False,
        **params
    ):
        obj = cls(params[projection], projection=projection)
        for key, value in params.items():
            if key != projection:
                if ignore_errors:
                    try:
                        obj.add_projection(key, value.points)
                    except Exception:
                        continue
                else:
                    obj.add_projection(key, value.points)
        return obj

    def get_projection(self, projection: str) -> Tensor:
        projections = self.projections
        if projection not in projections:
            raise KeyError(
                f"Projection '{projection}' not found. "
                f"Available projections: {set(projections.keys())}"
            )
        return projections[projection]

    def add_projection(
        self,
        projection: str,
        points: Tensor,
    ):
        if len(points) != self.n_points:
            raise ValueError(
                f'Number of points {len(points)} must match {self.n_points}.')
        self.point_data[f'_projection_{projection}'] = points

    def remove_projection(
        self,
        projection: str,
    ):
        projections = self.projections
        if projection not in projections:
            raise KeyError(
                f"Projection '{projection}' not found. "
                f"Available projections: {set(projections.keys())}"
            )
        del self.point_data[f'_projection_{projection}']

    def project(
        self,
        projection: str,
    ):
        projections = self.projections
        if projection not in projections:
            raise KeyError(
                f"Projection '{projection}' not found. "
                f"Available projections: {set(projections.keys())}"
            )
        self.points = projections[projection]


@dataclass
class CortexTriSurface:
    left: ProjectedPolyData
    right: ProjectedPolyData
    mask: Optional[str] = None

    @classmethod
    def from_darrays(
        cls,
        left: Mapping[str, Tuple[Tensor, Tensor]],
        right: Mapping[str, Tuple[Tensor, Tensor]],
        left_mask: Optional[Tensor] = None,
        right_mask: Optional[Tensor] = None,
        projection: Optional[str] = None,
    ):
        if projection is None:
            projection = list(left.keys())[0]
        left, mask_str = cls._hemisphere_darray_impl(
            left, left_mask, projection)
        right, _ = cls._hemisphere_darray_impl(
            right, right_mask, projection)
        return cls(left, right, mask_str)

    @classmethod
    def from_gifti(
        cls,
        left: Mapping[str, Union[str, nb.gifti.gifti.GiftiImage]],
        right: Mapping[str, Union[str, nb.gifti.gifti.GiftiImage]],
        left_mask: Optional[Union[str, nb.gifti.gifti.GiftiImage]] = None,
        right_mask: Optional[Union[str, nb.gifti.gifti.GiftiImage]] = None,
        projection: Optional[str] = None,
    ):
        left, left_mask = cls._hemisphere_gifti_impl(left, left_mask)
        right, right_mask = cls._hemisphere_gifti_impl(right, right_mask)
        return cls.from_darrays(
            left={k: tuple(d.data for d in v.darrays) for k, v in left.items()},
            right={k: tuple(d.data for d in v.darrays) for k, v in right.items()},
            left_mask=left_mask,
            right_mask=right_mask,
            projection=projection,
        )

    @classmethod
    def from_tflow(
        cls,
        template: str = 'fsLR',
        projections: Union[str, Sequence[str]] = 'veryinflated',
        load_mask: bool = False,
    ):
        if isinstance(projections, str):
            projections = (projections,)
        template = template_dict()[template]
        coor_query = template.TFLOW_COOR_QUERY
        lh, rh = {}, {}
        for projection in projections:
            coor_query.update(suffix=projection)
            lh_path, rh_path = (
                tflow.get(**coor_query, hemi='L'),
                tflow.get(**coor_query, hemi='R')
            )
            lh[projection] = lh_path
            rh[projection] = rh_path
        lh_mask, rh_mask = None, None
        if load_mask:
            mask_query = template.TFLOW_MASK_QUERY
            lh_mask, rh_mask = (
                tflow.get(**mask_query, hemi='L'),
                tflow.get(**mask_query, hemi='R')
            )
        return cls.from_gifti(lh, rh, lh_mask, rh_mask, projection=projections[0])

    @property
    def n_points(self) -> Mapping[str, int]:
        return {
            'left': self.left.n_points,
            'right': self.right.n_points,
        }

    @property
    def masks(self) -> Mapping[str, Tensor]:
        return {
            'left': self.left.point_data[self.mask],
            'right': self.right.point_data[self.mask],
        }

    @property
    def mask_size(self) -> Mapping[str, int]:
        return {
            'left': self.masks['left'].sum().item(),
            'right': self.masks['right'].sum().item(),
        }

    @staticmethod
    def _hemisphere_darray_impl(data, mask, projection):
        surf = pv.make_tri_mesh(*data[projection])
        surf = ProjectedPolyData(surf, projection=projection)
        for key, (points, _) in data.items():
            if key != projection:
                surf.add_projection(key, points)
        mask_str = None
        if mask is not None:
            surf.point_data['__mask__'] = mask
            mask_str = '__mask__'
        return surf, mask_str

    @staticmethod
    def _hemisphere_gifti_impl(data, mask):
        data = {
            k: (nb.load(v) if is_path_like(v) else v)
            for k, v in data.items()}
        if mask is not None:
            if is_path_like(mask):
                mask = nb.load(mask)
            mask = mask.darrays[0].data.astype(bool)
        return data, mask
