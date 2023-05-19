# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Brain surfaces
~~~~~~~~~~~~~~
Brain surface objects for plotting.
"""
import pathlib
import warnings
import pyvista as pv
import numpy as np
import nibabel as nb
import templateflow.api as tflow

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Union
from matplotlib.colors import ListedColormap
from hypercoil.engine import Tensor
from hypercoil.functional.sphere import (
    _euc_dist,
    spherical_geodesic,
)
from hypercoil.init.atlasmixins import Reference
from hypercoil.neuro.const import (
    CIfTIStructures,
    template_dict,
    neuromaps_fetch_fn,
)


def is_path_like(obj: Any) -> bool:
    return isinstance(obj, (str, pathlib.Path))


def extend_to_max(*pparams, axis=-1):
    max_len = max(p.shape[axis] for p in pparams)
    pad = [(0, 0)] * pparams[0].ndim
    for p in pparams:
        pad[axis] = (0, max_len - p.shape[axis])
        yield np.pad(p, pad, 'constant')


# class ProjectionKey:
#     """
#     Currently, PyVista does not support non-string keys for point data. This
#     class is unused, but is a placeholder in case PyVista supports this in
#     the future.
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
        projection: str = None,
        **params
    ):
        super().__init__(*pparams, **params)
        self.point_data[f'_projection_{projection}'] = self.points.copy()

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
            left={k: tuple(d.data for d in v.darrays)
                  for k, v in left.items()},
            right={k: tuple(d.data for d in v.darrays)
                   for k, v in right.items()},
            left_mask=left_mask,
            right_mask=right_mask,
            projection=projection,
        )

    @classmethod
    def _from_archive(
        cls,
        fetch_fn: callable,
        coor_query: dict,
        mask_query: dict,
        projections: Union[str, Sequence[str]],
        load_mask: bool,
    ):
        if isinstance(projections, str):
            projections = (projections,)
        lh, rh = {}, {}
        for projection in projections:
            coor_query.update(suffix=projection)
            lh_path, rh_path = (
                fetch_fn(**coor_query, hemi="L"),
                fetch_fn(**coor_query, hemi="R"),
            )
            lh[projection] = lh_path
            rh[projection] = rh_path
        lh_mask, rh_mask = None, None
        if load_mask:
            lh_mask, rh_mask = (
                fetch_fn(**mask_query, hemi="L"),
                fetch_fn(**mask_query, hemi="R")
            )
        return cls.from_gifti(lh, rh, lh_mask, rh_mask,
                              projection=projections[0])

    @classmethod
    def from_tflow(
        cls,
        template: str = 'fsLR',
        projections: Union[str, Sequence[str]] = 'veryinflated',
        load_mask: bool = False,
    ):
        template = template_dict()[template]
        return cls._from_archive(
            fetch_fn=tflow.get,
            coor_query=template.TFLOW_COOR_QUERY,
            mask_query=template.TFLOW_MASK_QUERY,
            projections=projections,
            load_mask=load_mask,
        )

    @classmethod
    def from_nmaps(
        cls,
        template: str = 'fsaverage',
        projections: Union[str, Sequence[str]] = 'pial',
        load_mask: bool = False,
    ):
        template = template_dict()[template]
        return cls._from_archive(
            fetch_fn=neuromaps_fetch_fn,
            coor_query=template.NMAPS_COOR_QUERY,
            mask_query=template.NMAPS_MASK_QUERY,
            projections=projections,
            load_mask=load_mask,
        )

    @property
    def n_points(self) -> Mapping[str, int]:
        return {
            'left': self.left.n_points,
            'right': self.right.n_points,
        }

    @property
    def point_data(self) -> Mapping[str, Mapping]:
        return {
            'left': self.left.point_data,
            'right': self.right.point_data,
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

    def add_cifti_dataset(
        self,
        name: str,
        cifti: Union[str, nb.cifti2.cifti2.Cifti2Image],
        is_masked: bool = False,
        apply_mask: bool = True,
        null_value: Optional[float] = 0.,
    ):
        names_dict = {
            CIfTIStructures.LEFT : 'left',
            CIfTIStructures.RIGHT : 'right',
        }
        slices = {}

        if is_path_like(cifti):
            cifti = nb.load(cifti)
        ref = Reference(cifti, model_axes='cifti')
        model_axis = ref.model_axobj

        offset = 0
        for struc, slc, _ in (model_axis.iter_structures()):
            hemi = names_dict.get(struc, None)
            if hemi is not None:
                start, stop = slc.start, slc.stop
                start = start if start is not None else 0
                stop = (
                    stop if stop is not None
                    else offset + self.mask_size[hemi]
                )
                slices[hemi] = slice(start, stop)
                offset = stop

        data = ref.dataobj
        while data.shape[0] == 1:
            data = data[0]
        return self.add_vertex_dataset(
            name=name,
            data=data,
            left_slice=slices['left'],
            right_slice=slices['right'],
            default_slices=False,
            is_masked=is_masked,
            apply_mask=apply_mask,
            null_value=null_value,
        )

    def add_gifti_dataset(
        self,
        name: str,
        left_gifti: Optional[Union[str, nb.gifti.gifti.GiftiImage]] = None,
        right_gifti: Optional[Union[str, nb.gifti.gifti.GiftiImage]] = None,
        is_masked: bool = False,
        apply_mask: bool = True,
        null_value: Optional[float] = 0.,
        map_all: bool = True,
        arr_idx: int = 0,
        select: Optional[Sequence[int]] = None,
        exclude: Optional[Sequence[int]] = None,
    ):
        left_data = left_gifti.darrays if left_gifti else []
        right_data = right_gifti.darrays if right_gifti else []
        if map_all and len(left_data) > 1 and len(right_data) > 1:
            if left_data and right_data and len(left_data) != len(right_data):
                raise ValueError(
                    "Left and right hemisphere gifti images must have the "
                    "same number of data arrays."
                )
            n_darrays = max(len(left_data), len(right_data))
            exclude = exclude or []
            if select is not None:
                exclude = [i for i in range(n_darrays) if i not in select]
            for i in range(n_darrays):
                if i in exclude:
                    continue
                name_i = f"{name}_{i}"
                data_l = left_data[i].data if left_gifti else None
                data_r = right_data[i].data if right_gifti else None
                self.add_vertex_dataset(
                    name=name_i,
                    left_data=data_l,
                    right_data=data_r,
                    is_masked=is_masked,
                    apply_mask=apply_mask,
                    null_value=null_value,
                )
        else:
            left_data = left_data[arr_idx].data if left_gifti else None
            right_data = right_data[arr_idx].data if right_gifti else None
            self.add_vertex_dataset(
                name=name,
                left_data=left_data,
                right_data=right_data,
                is_masked=is_masked,
                apply_mask=apply_mask,
                null_value=null_value,
            )

    def add_vertex_dataset(
        self,
        name: str,
        data: Optional[Tensor] = None,
        left_data: Optional[Tensor] = None,
        right_data: Optional[Tensor] = None,
        left_slice: Optional[slice] = None,
        right_slice: Optional[slice] = None,
        default_slices: bool = True,
        is_masked: bool = False,
        apply_mask: bool = True,
        null_value: Optional[float] = 0.,
    ):
        if data is not None:
            if left_slice is None and right_slice is None:
                if default_slices:
                    if is_masked:
                        left_slice = slice(0, self.mask_size['left'])
                        right_slice = slice(self.mask_size['left'], None)
                    else:
                        left_slice = slice(0, self.n_points['left'])
                        right_slice = slice(self.n_points['left'], None)
                else:
                    warnings.warn(
                        "No slices were provided for vertex data, and default "
                        "slicing was toggled off. The `data` tensor will be "
                        "ignored. Attempting fallback to `left_data` and "
                        "`right_data`.\n\n"
                        "To silence this warning, provide slices for the data "
                        "or toggle on default slicing if a `data` tensor is "
                        "provided. Alternatively, provide `left_data` and "
                        "`right_data` directly and exclusively."
                    )
            if left_slice is not None:
                if left_data is not None:
                    warnings.warn(
                        "Both `left_data` and `left_slice` were provided. "
                        "The `left_data` tensor will be ignored. "
                        "To silence this warning, provide `left_data` or "
                        "`left_slice` exclusively."
                    )
                left_data = data[..., left_slice]
            if right_slice is not None:
                if right_data is not None:
                    warnings.warn(
                        "Both `right_data` and `right_slice` were provided. "
                        "The `right_data` tensor will be ignored. "
                        "To silence this warning, provide `right_data` or "
                        "`right_slice` exclusively."
                    )
                right_data = data[..., right_slice]
        if left_data is None and right_data is None:
            raise ValueError(
                "Either no data was provided, or insufficient information "
                "was provided to slice the data into left and right "
                "hemispheres.")
        if left_data is not None:
            self.left.point_data[name] = self._hemisphere_vertex_data_impl(
                left_data, is_masked, apply_mask, null_value, 'left',
            )
        if right_data is not None:
            self.right.point_data[name] = self._hemisphere_vertex_data_impl(
                right_data, is_masked, apply_mask, null_value, 'right',
            )

    def parcellate_vertex_dataset(
        self,
        name: str,
        parcellation: str,
    ) -> Tensor:
        parcellation_left = self._hemisphere_parcellate_impl(
            name, parcellation, 'left',
        )
        parcellation_right = self._hemisphere_parcellate_impl(
            name, parcellation, 'right',
        )
        parcellation_left, parcellation_right = extend_to_max(
            parcellation_left,
            parcellation_right,
            axis=0)
        return parcellation_left + parcellation_right

    def scatter_into_parcels(
        self,
        data: Tensor,
        parcellation: str,
        sink: Optional[str] = None
    ) -> Tensor:
        scattered_left = self._hemisphere_into_parcels_impl(
            data, parcellation, 'left',
        )
        scattered_right = self._hemisphere_into_parcels_impl(
            data, parcellation, 'right',
        )
        if sink is not None:
            self.left.point_data[sink] = scattered_left
            self.right.point_data[sink] = scattered_right
        return (scattered_left, scattered_right)

    def parcel_centres_of_mass(
        self,
        parcellation: str,
        projection: str,
    ):
        projection_name = f"_projection_{projection}"
        return self.parcellate_vertex_dataset(
            projection_name,
            parcellation=parcellation
        )

    def scalars_centre_of_mass(
        self,
        hemisphere: str,
        scalars: str,
        projection: str,
    ) -> Tensor:
        projection_name = f"_projection_{projection}"
        if hemisphere == 'left':
            proj_data = self.left.point_data[projection_name]
            scalars_data = self.left.point_data[scalars].reshape(-1, 1)
        elif hemisphere == 'right':
            proj_data = self.right.point_data[projection_name]
            scalars_data = self.right.point_data[scalars].reshape(-1, 1)
        else:
            raise ValueError(
                f"Invalid hemisphere: {hemisphere}. Must be 'left' or 'right'.")
        num = np.nansum(proj_data * scalars_data, axis=0)
        den = np.nansum(scalars_data, axis=0)
        return num / den

    def scalars_peak(
        self,
        hemisphere: str,
        scalars: str,
        projection: str
    ) -> Tensor:
        projection_name = f"_projection_{projection}"
        if hemisphere == 'left':
            proj_data = self.left.point_data[projection_name]
            scalars_data = self.left.point_data[scalars].reshape(-1, 1)
        elif hemisphere == 'right':
            proj_data = self.right.point_data[projection_name]
            scalars_data = self.right.point_data[scalars].reshape(-1, 1)
        else:
            raise ValueError(
                f"Invalid hemisphere: {hemisphere}. Must be 'left' or 'right'.")
        return proj_data[np.argmax(scalars_data)]

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

    def _hemisphere_vertex_data_impl(
        self,
        data: Tensor,
        is_masked: bool,
        apply_mask: bool,
        null_value: Optional[float],
        hemisphere: str,
    ) -> Tensor:
        if null_value is None:
            null_value = np.nan
        if is_masked:
            if data.ndim == 2:
                data = data.T
                init = np.full((self.n_points[hemisphere], data.shape[-1]), null_value)
            else:
                init = np.full(self.n_points[hemisphere], null_value)
            init[self.masks[hemisphere]] = data
            return init
        elif apply_mask:
            if data.ndim == 2:
                data = data.T
                mask = self.masks[hemisphere][..., None]
            else:
                mask = self.masks[hemisphere]
            return np.where(
                mask,
                data,
                null_value,
            )
        else:
            return data

    @staticmethod
    def _hemisphere_parcellation_impl(
        point_data: Mapping,
        parcellation: str,
        null_value: Optional[float] = 0.,
    ) -> Tensor:
        parcellation = point_data[parcellation]
        if null_value is None:
            null_value = parcellation.min() - 1
        parcellation[np.isnan(parcellation)] = null_value
        parcellation = parcellation.astype(int)
        if parcellation.ndim == 1:
            parcellation = parcellation - parcellation.min()
            null_index = int(null_value - parcellation.min())
            n_parcels = parcellation.max() + 1
            parcellation = np.eye(n_parcels)[parcellation]
            parcellation = np.delete(parcellation, null_index, axis=1)
        denom = parcellation.sum(axis=0, keepdims=True)
        denom = np.where(denom == 0, 1, denom)
        return parcellation / denom

    def _hemisphere_parcellate_impl(
        self,
        name: str,
        parcellation: str,
        hemisphere: str,
    ) -> Tensor:
        try:
            point_data = self.point_data[hemisphere]
        except KeyError:
            return None
        parcellation = self._hemisphere_parcellation_impl(
            point_data,
            parcellation=parcellation,
        )
        return parcellation.T @ point_data[name]

    def _hemisphere_into_parcels_impl(
        self,
        data: Tensor,
        parcellation: str,
        hemisphere: str,
    ) -> Tensor:
        try:
            point_data = self.point_data[hemisphere]
        except KeyError:
            return None
        parcellation = self._hemisphere_parcellation_impl(
            point_data,
            parcellation=parcellation,
        )
        return parcellation @ data[:parcellation.shape[-1]]

    def poles(self, hemisphere: str) -> Tensor:
        if hemisphere == "left":
            pole_names = (
                "lateral", "posterior", "ventral",
                "medial", "anterior", "dorsal"
            )
        elif hemisphere == "right":
            pole_names = (
                "medial", "posterior", "ventral",
                "lateral", "anterior", "dorsal"
            )
        coors = self.__getattribute__(hemisphere).points
        pole_index = np.concatenate((
            coors.argmin(axis=0),
            coors.argmax(axis=0)
        ))
        poles = coors[pole_index]
        return dict(zip(pole_names, poles))

    def closest_poles(
        self,
        hemisphere: str,
        coors: Tensor,
        metric: str = "euclidean",
        n_poles: int = 1,
    ) -> Tensor:
        poles = self.poles(hemisphere)
        poles_coors = np.array(list(poles.values()))
        if metric == "euclidean":
            dists = _euc_dist(coors, poles_coors)
        elif metric == "spherical":
            proj = self.projection
            self.__getattribute__(hemisphere).project("sphere")
            dists = spherical_geodesic(coors, poles_coors)
            self.__getattribute__(hemisphere).project(proj)
        #TODO: add case for the geodesic on the manifold as implemented in
        #      PyVista/VTK.
        else:
            raise ValueError(
                f"Metric must currently be either euclidean or spherical, "
                f"but {metric} was given.")
        pole_index = np.argsort(dists, axis=1)[:, :n_poles]
        pole_names = np.array(list(poles.keys()))
        #TODO: need to use tuple indexing here
        return pole_names[pole_index]


def _cmap_impl_hemisphere(
    surf: CortexTriSurface,
    hemisphere: Literal['left', 'right'],
    parcellation: str,
    colours: Tensor,
    null_value: float
) -> Tuple[Tensor, Tuple[float, float]]:
    parcellation = surf.point_data[hemisphere][parcellation]
    start = int(np.min(parcellation[parcellation != null_value])) - 1
    stop = int(np.max(parcellation))
    cmap = ListedColormap(colours[start:stop, :3])
    clim = (start + 0.1, stop + 0.9)
    return cmap, clim


def make_cmap(surf, cmap, parcellation, null_value=0, separate=True):
    colours = surf.parcellate_vertex_dataset(cmap, parcellation)
    colours = np.minimum(colours, 1)
    colours = np.maximum(colours, 0)

    cmap_left, clim_left = _cmap_impl_hemisphere(
        surf,
        'left',
        parcellation,
        colours,
        null_value
    )
    cmap_right, clim_right = _cmap_impl_hemisphere(
        surf,
        'right',
        parcellation,
        colours,
        null_value
    )
    if separate:
        return (cmap_left, clim_left), (cmap_right, clim_right)
    else:
        #TODO: Rewrite this to skip the unnecessary intermediate blocks above.
        cmin = min(clim_left[0], clim_right[0])
        cmax = max(clim_left[1], clim_right[1])
        cmap = ListedColormap(colours[:, :3])
        clim = (cmin, cmax)
        return cmap, clim
