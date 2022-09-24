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
        self.projection = projection
        self.projections = {
            projection: self.points,
        }

    def __repr__(self):
        return super().__repr__()

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

    def add_projection(
        self,
        projection: str,
        points: Tensor,
    ):
        if len(points) != self.n_points:
            raise ValueError(
                f'Number of points {len(points)} must match {self.n_points}.')
        self.projections[projection] = points

    def remove_projection(
        self,
        projection: str,
    ):
        if projection not in self.projections:
            raise KeyError(f'Projection {projection} not found.')
        del self.projections[projection]

    def project(
        self,
        projection: str,
    ):
        if projection not in self.projections:
            raise KeyError(f'Projection {projection} not found.')
        self.points = self.projections[projection]


@dataclass
class BrainTriSurface:
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
        left, mask_str = cls._hemisphere_darray(
            left, left_mask, projection)
        right, _ = cls._hemisphere_darray(
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
        left, left_mask = cls._hemisphere_gifti(left, left_mask)
        right, right_mask = cls._hemisphere_gifti(right, right_mask)
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

    @staticmethod
    def _hemisphere_darray(data, mask, projection):
        surf = pv.make_tri_mesh(*data[projection])
        surf = ProjectedPolyData(surf, projection=projection)
        for key, value in data.items():
            if key != projection:
                surf.add_projection(key, value[0])
        mask_str = None
        if mask is not None:
            surf.point_data['__mask__'] = mask
            mask_str = '__mask__'
        return surf, mask_str

    @staticmethod
    def _hemisphere_gifti(data, mask):
        print(data)
        data = {
            k: (nb.load(v) if is_path_like(v) else v)
            for k, v in data.items()}
        if mask is not None:
            if is_path_like(mask):
                mask = nb.load(mask)
            mask = mask.darrays[0].data.astype(bool)
        return data, mask

    def add_scalar(
        self,
        name: str,
        left: Optional[Tensor] = None,
        right: Optional[Tensor] = None,
        apply_mask: bool = True,
    ):
        if left is not None:
            if apply_mask and self.mask is not None:
                left = left[self.left.point_data[self.mask]]
            self.left.point_data[name] = left
        if right is not None:
            if apply_mask and self.mask is not None:
                right = right[self.right.point_data[self.mask]]
            self.right.point_data[name] = right
