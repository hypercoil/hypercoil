# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Unified plotter
~~~~~~~~~~~~~~~
Unified plotting function for surface, volume, and network data.
"""
import dataclasses
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple, Union

from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import pyvista as pv

from .surf import (
    CortexTriSurface,
    ProjectedPolyData,
)
from .utils import cortex_theme, robust_clim


@dataclasses.dataclass
class HemisphereParameters:
    left: Mapping[str, Any]
    right: Mapping[str, Any]

    def get(self, hemi, param):
        return self.__getattribute__(hemi)[param]


def _get_hemisphere_parameters(
    *,
    surf_scalars_cmap: Any,
    surf_scalars_clim: Any,
) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    left = {}
    right = {}
    def assign_tuple(arg, name):
        l, r = arg
        left[name] = l
        right[name] = r
    def assign_scalar(arg, name):
        left[name] = right[name] = arg
    def conditional_assign(condition, arg, name):
        if condition(arg):
            assign_tuple(arg, name)
        else:
            assign_scalar(arg, name)

    conditional_assign(
        lambda x: len(x) == 2,
        surf_scalars_cmap,
        'surf_scalars_cmap'
    )
    conditional_assign(
        lambda x: len(x) == 2 and isinstance(x[0], (tuple, list)),
        surf_scalars_clim,
        'surf_scalars_clim',
    )
    return HemisphereParameters(left, right)


def unified_plotter(
    *,
    surf: Optional["CortexTriSurface"] = None,
    surf_projection: str = "pial",
    surf_alpha: float = 1.0,
    surf_scalars: Optional[str] = None,
    surf_scalars_boundary_color: str = "black",
    surf_scalars_boundary_width: int = 0,
    surf_scalars_cmap: Any = (None, None),
    surf_scalars_clim: Any = "robust",
    surf_scalars_below_color: str = "black",
    vol_coor: Optional[np.ndarray] = None,
    vol_scalars: Optional[np.ndarray] = None,
    vol_scalars_point_size: Optional[float] = None,
    vol_voxdim: Optional[Sequence[float]] = None,
    vol_scalars_cmap: Optional[str] = None,
    vol_scalars_clim: Optional[tuple] = None,
    edge_values: Optional[pd.DataFrame] = None,
    node_values: Optional[pd.DataFrame] = None,
    node_coor: Optional[np.ndarray] = None,
    node_parcel_scalars: Optional[str] = None,
    node_color: Optional[str] = "black",
    node_radius: Union[float, str] = 3.0,
    node_radius_range: Tuple[float, float] = (2, 10),
    node_cmap: Any = "viridis",
    node_cmap_range: Tuple[float, float] = (0, 1),
    node_alpha: Union[float, str] = 1.0,
    node_lh: Optional[np.ndarray] = None,
    edge_color: Optional[str] = "edge_sgn",
    edge_radius: Union[float, str] = "edge_val",
    edge_radius_range: Tuple[float, float] = (0.1, 1.8),
    edge_cmap: Any = "RdBu_r",
    edge_cmap_range: Tuple[float, float] = (0, 1),
    edge_alpha: Union[float, str] = 1.0,
    hemisphere: Optional[Literal["left", "right"]] = None,
    hemisphere_slack: float = 0.,
    off_screen: bool = True,
    copy_actors: bool = False,
    theme: Optional[pv.themes.DocumentTheme] = None,
    views: Sequence = (),
    return_plotter: bool = False,
    return_screenshot: bool = True,
    return_html: bool = False,
) -> Optional[Sequence[pv.Plotter]]:

    # Helper functions for graph plots
    def get_node_color(color):
        if (
            isinstance(color, str) or
            isinstance(color, tuple) or
            isinstance(color, list)
        ):
            return color
        else:
            try:
                cmap = get_cmap(node_cmap)
            except ValueError:
                cmap = node_cmap
            vmin, vmax = node_cmap_range
            norm = Normalize(vmin=vmin, vmax=vmax)
            return cmap(norm(color))

    def get_edge_color(color):
        if (
            isinstance(color, str) or
            isinstance(color, tuple) or
            isinstance(color, list)
        ):
            return color
        else:
            try:
                cmap = get_cmap(edge_cmap)
            except ValueError:
                cmap = edge_cmap
            vmin, vmax = edge_cmap_range
            norm = Normalize(vmin=vmin, vmax=vmax)
            return cmap(norm(color))

    def process_edge_values():
        start_node, end_node = tuple(zip(*edge_values.index))
        start_node, end_node = np.array(start_node), np.array(end_node)
        start = node_coor[start_node]
        end = node_coor[end_node]
        centre = (start + end) / 2
        direction = end - start
        length = np.linalg.norm(direction, axis=-1)
        direction = direction / length.reshape(-1, 1)
        return centre, direction, length

    def map_to_attr(values, attr, attr_range):
        if attr == "index":
            attr = np.array(values.index)
        else:
            attr = values[attr]
        if attr_range is None:
            return attr
        max_val = attr.max()
        min_val = attr.min()
        return (
            attr_range[0]
            + (attr_range[1] - attr_range[0])
            * (attr - min_val)
            / (max_val - min_val)
        )

    def map_to_radius():
        if isinstance(node_radius, float):
            return (node_radius,) * len(node_values)
        else:
            return map_to_attr(node_values, node_radius, node_radius_range)

    def map_to_color():
        if node_color in node_values.columns or node_color == "index":
            return map_to_attr(node_values, node_color, node_cmap_range)
        else:
            return (node_color,) * len(node_values)

    def map_to_opacity():
        if isinstance(node_alpha, float):
            return (node_alpha,) * len(node_values)
        #TODO: this will fail if you want to use the index as the opacity.
        #      There's no legitimate reason you would want to do this
        #      so it's a very low priority fix.
        opa_min = max(0, node_values[node_alpha].min())
        opa_max = min(1, node_values[node_alpha].max())
        return map_to_attr(node_values, node_alpha, (opa_min, opa_max))

    def map_edge_to_radius():
        if isinstance(edge_radius, float):
            return (edge_radius,) * len(edge_values)
        else:
            return map_to_attr(edge_values, edge_radius, edge_radius_range)

    def map_edge_to_color():
        if edge_color in edge_values.columns or edge_color == "index":
            return map_to_attr(edge_values, edge_color, edge_cmap_range)
        else:
            return (edge_color,) * len(edge_values)

    def map_edge_to_opacity():
        if isinstance(edge_alpha, float):
            return (edge_alpha,) * len(edge_values)
        #TODO: this will fail if you want to use the index as the opacity.
        #      There's no conceivable reason you would want to do this
        #      (and besides, it's a multiindex...) so it's a very low
        #      priority fix.
        opa_min = max(0, edge_values[edge_alpha].min())
        opa_max = min(1, edge_values[edge_alpha].max())
        return map_to_attr(edge_values, edge_alpha, (opa_min, opa_max))    

    #TODO: cortex_theme doesn't work here for some reason. If the background
    #      is transparent, all of the points are also made transparent. So
    #      we're sticking with a white background for now.
    theme = theme or pv.themes.DocumentTheme()

    hemispheres = (
        (hemisphere,) if hemisphere is not None
        else ("left", "right")
    )
    hemi_params = _get_hemisphere_parameters(
        surf_scalars_cmap=surf_scalars_cmap,
        surf_scalars_clim=surf_scalars_clim,
    )
    if node_parcel_scalars is not None:
        node_coor = surf.parcel_centres_of_mass(
            node_parcel_scalars,
            surf_projection,
        )

    p = pv.Plotter(off_screen=off_screen, theme=theme)

    if surf is not None:
        for hemisphere in hemispheres:
            hemi_surf = surf.__getattribute__(hemisphere)
            hemi_surf.project(surf_projection)
            hemi_clim = hemi_params.get(hemisphere, 'surf_scalars_clim')
            hemi_cmap = hemi_params.get(hemisphere, 'surf_scalars_cmap')
            hemi_color = None if hemi_cmap else 'white'
            if hemi_clim == "robust" and surf_scalars is not None:
                hemi_clim = robust_clim(hemi_surf, surf_scalars)
            #TODO: copying the mesh seems like it could create memory issues.
            #      A better solution would be delayed execution.
            p.add_mesh(
                hemi_surf,
                opacity=surf_alpha,
                show_edges=False,
                scalars=surf_scalars,
                cmap=hemi_cmap,
                clim=hemi_clim,
                color=hemi_color,
                below_color=surf_scalars_below_color,
                copy_mesh=copy_actors,
            )
            if surf_scalars_boundary_width > 0 and surf_scalars is not None:
                p.add_mesh(
                    hemi_surf.contour(
                        isosurfaces=range(
                            int(max(hemi_surf.point_data[surf_scalars]))
                        )
                    ),
                    color=surf_scalars_boundary_color,
                    line_width=surf_scalars_boundary_width,
                )

    if vol_scalars is not None:
        assert vol_coor is not None, (
            "Volumetric scalars provided with unspecified coordinates")
        vol_scalars_point_size = vol_scalars_point_size or min(vol_voxdim[:3])
        p.add_points(
            vol_coor.T,
            render_points_as_spheres=False,
            style='points_gaussian',
            emissive=False,
            scalars=vol_scalars,
            opacity=0.99,
            point_size=vol_scalars_point_size,
            ambient=1.0,
            cmap=vol_scalars_cmap,
            clim=vol_scalars_clim,
        )

    if node_coor is not None:
        for c, col, rad, opa in zip(
            node_coor,
            map_to_color(),
            map_to_radius(),
            map_to_opacity(),
        ):
            node = pv.Icosphere(
                radius=rad,
                center=c,
            )
            p.add_mesh(
                node,
                color=get_node_color(col),
                opacity=opa,
            )
    if edge_values is not None:
        for c, d, ht, col, rad, opa in zip(
            *process_edge_values(),
            map_edge_to_color(),
            map_edge_to_radius(),
            map_edge_to_opacity(),
        ):
            edge = pv.Cylinder(
                center=c,
                direction=d,
                height=ht,
                radius=rad,
            )
            p.add_mesh(
                edge,
                color=get_edge_color(col),
                opacity=opa,
            )

    return p
