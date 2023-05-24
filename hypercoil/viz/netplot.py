# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Brain network plotting
~~~~~~~~~~~~~~~~~~~~~~
Brain network plotting utilities.
"""
from typing import Any, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import pyvista as pv
import pandas as pd
from matplotlib.cm import get_cmap
from matplotlib.colors import Normalize

from .surf import CortexTriSurface
from .utils import cortex_theme


def filter_node_data(
    val: np.ndarray,
    name: str = "node",
    threshold: Union[float, int] = 0.0,
    percent_threshold: bool = False,
    topk_threshold: bool = False,
    absolute: bool = True,
    node_selection: Optional[np.ndarray] = None,
    incident_edge_selection: Optional[np.ndarray] = None,
    removed_val: Optional[float] = None,
    surviving_val: Optional[float] = 1.0,
) -> pd.DataFrame:
    node_incl = np.ones_like(val, dtype=bool)

    sgn = np.sign(val)
    if absolute:
        val = np.abs(val)
    if node_selection is not None:
        node_incl[~node_selection] = 0
    if incident_edge_selection is not None:
        node_incl[~incident_edge_selection.any(axis=-1)] = 0
    if topk_threshold:
        indices = np.argpartition(val, int(threshold))
        node_incl[indices[int(threshold):]] = 0
    elif percent_threshold:
        node_incl[val < np.percentile(val[node_incl], 100 * threshold)] = 0
    else:
        node_incl[val < threshold] = 0

    if removed_val is not None:
        val[~node_incl] = removed_val
        if surviving_val is not None:
            val[node_incl] = surviving_val
        index = np.arange(val.shape[0])
    else:
        val = val[node_incl]
        sgn = sgn[node_incl]
        index = np.where(node_incl)[0]

    return pd.DataFrame({
        f"{name}_val": val,
        f"{name}_sgn": sgn,
    }, index=pd.Index(index, name="node"))


def filter_adjacency_data(
    adj: np.ndarray,
    name: str = "edge",
    threshold: Union[float, int] = 0.0,
    percent_threshold: bool = False,
    topk_threshold_nodewise: bool = False,
    absolute: bool = True,
    incident_node_selection: Optional[np.ndarray] = None,
    connected_node_selection: Optional[np.ndarray] = None,
    edge_selection: Optional[np.ndarray] = None,
    removed_val: Optional[float] = None,
    surviving_val: Optional[float] = 1.0,
    emit_degree: Union[bool, Literal["abs", "+", "-"]] = False,
) -> pd.DataFrame:
    adj_incl = np.ones_like(adj, dtype=bool)

    sgn = np.sign(adj)
    if absolute:
        adj = np.abs(adj)
    if incident_node_selection is not None:
        adj_incl[~incident_node_selection, :] = 0
    if connected_node_selection is not None:
        adj_incl[~connected_node_selection, :] = 0
        adj_incl[:, ~connected_node_selection] = 0
    if edge_selection is not None:
        adj_incl[~edge_selection] = 0
    if topk_threshold_nodewise:
        indices = np.argpartition(adj, int(threshold), axis=-1)
        indices = indices[..., int(threshold):]
        adj_incl[np.arange(adj.shape[0], dtype=int).reshape(-1, 1), indices] = 0
    elif percent_threshold:
        adj_incl[adj < np.percentile(adj[adj_incl], 100 * threshold)] = 0
    else:
        adj_incl[adj < threshold] = 0

    degree = None
    if emit_degree == "abs":
        degree = np.abs(adj).sum(axis=0)
    elif emit_degree == "+":
        degree = np.maximum(adj, 0).sum(axis=0)
    elif emit_degree == "-":
        degree = -np.minimum(adj, 0).sum(axis=0)
    elif emit_degree:
        degree = adj.sum(axis=0)

    indices_incl = np.triu_indices(adj.shape[0], k=1)
    adj_incl = adj_incl | adj_incl.T

    if removed_val is not None:
        adj[~adj_incl] = removed_val
        if surviving_val is not None:
            adj[adj_incl] = surviving_val
    else:
        adj_incl = adj_incl[indices_incl]
        indices_incl = tuple(i[adj_incl] for i in indices_incl)
    adj = adj[indices_incl]
    sgn = sgn[indices_incl]

    edge_values = pd.DataFrame({
        f"{name}_val": adj,
        f"{name}_sgn": sgn,
    }, index=pd.MultiIndex.from_arrays(indices_incl, names=["src", "dst"]))

    if degree is not None:
        degree = pd.DataFrame(
            degree,
            index=range(1, degree.shape[0] + 1),
            columns=(f"{name}_degree",)
        )
        return edge_values, degree
    return edge_values


def embedded_graph_plotter(
    *,
    surf: "CortexTriSurface",
    edge_values: pd.DataFrame,
    node_values: Optional[pd.DataFrame] = None,
    coor: Optional[np.ndarray] = None,
    parcellation: Optional[str] = None,
    node_color: Optional[str] = "black",
    node_radius: Union[float, str] = 3.0,
    node_cmap: Any = "viridis",
    node_cmap_range: Tuple[float, float] = (0, 1),
    node_radius_range: Tuple[float, float] = (2, 10),
    edge_color: Optional[str] = "edge_sgn",
    edge_radius: Union[float, str] = "edge_val",
    edge_cmap: Any = "coolwarm",
    edge_cmap_range: Tuple[float, float] = (0, 1),
    edge_radius_range: Tuple[float, float] = (0.1, 1.8),
    projection: Optional[str] = "pial",
    surf_opacity: float = 0.1,
    hemisphere_slack: float = 1.1,
    node_lh: Optional[np.ndarray] = None,
    off_screen: bool = True,
    theme: Optional[pv.themes.DocumentTheme] = None,
) -> pv.Plotter:
    def get_node_color(color):
        if isinstance(color, str) or isinstance(color, tuple) or isinstance(color, list):
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
        if isinstance(color, str) or isinstance(color, tuple) or isinstance(color, list):
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
        start = coor[start_node]
        end = coor[end_node]
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

    theme = theme or cortex_theme()
    p = pv.Plotter(off_screen=off_screen, theme=theme)
    if parcellation is not None:
        coor = surf.parcel_centres_of_mass(
            parcellation,
            projection,
        )
    assert coor is not None, "Either coor or parcellation must be provided"

    surf.left.project(projection)
    surf.right.project(projection)

    if node_lh is not None:
        hw_left = (surf.left.bounds[1] - surf.left.bounds[0]) / 2
        hw_right = (surf.right.bounds[1] - surf.right.bounds[0]) / 2
        hemi_gap = surf.right.center[0] - surf.left.center[0]
        min_gap = hw_left + hw_right
        target_gap = min_gap * hemisphere_slack
        displacement = (target_gap - hemi_gap) / 2
        l = surf.left.translate((-displacement, 0, 0))
        r = surf.right.translate((displacement, 0, 0))
        coor[node_lh, 0] -= displacement
        coor[~node_lh, 0] += displacement

    p.add_mesh(
        l,
        opacity=surf_opacity,
        show_edges=False,
        color='white',
    )
    p.add_mesh(
        r,
        opacity=surf_opacity,
        show_edges=False,
        color='white',
    )
    for c, col, rad in zip(coor, map_to_color(), map_to_radius()):
        node = pv.Icosphere(
            radius=rad,
            center=c,
        )
        p.add_mesh(
            node,
            color=get_node_color(col),
            opacity=1.0,
        )
    for c, d, ht, col, rad in zip(
        *process_edge_values(),
        map_edge_to_color(),
        map_edge_to_radius()
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
            opacity=1.0,
        )
    return p


def plot_embedded_graph(
    *,
    surf: "CortexTriSurface",
    edge_values: pd.DataFrame,
    node_lh: np.ndarray,
    node_values: Optional[pd.DataFrame] = None,
    coor: Optional[np.ndarray] = None,
    parcellation: Optional[str] = None,
    node_color: Optional[str] = "black",
    node_radius: Union[float, str] = 3.0,
    node_cmap: Any = "viridis",
    node_cmap_range: Tuple[float, float] = (0, 1),
    node_radius_range: Tuple[float, float] = (2, 10),
    edge_color: Optional[str] = "edge_sgn",
    edge_radius: Union[float, str] = "edge_val",
    edge_cmap: Any = "coolwarm",
    edge_cmap_range: Tuple[float, float] = (0, 1),
    edge_radius_range: Tuple[float, float] = (0.1, 1.8),
    projection: Optional[str] = "pial",
    surf_opacity: float = 0.1,
    hemisphere_slack: float = 1.1,
    off_screen: bool = True,
    theme: Optional[pv.themes.DocumentTheme] = None,
    **params,
) -> Mapping:
    p = embedded_graph_plotter(
        surf=surf,
        edge_values=edge_values,
        node_lh=node_lh,
        node_values=node_values,
        coor=coor,
        parcellation=parcellation,
        node_color=node_color,
        node_radius=node_radius,
        node_cmap=node_cmap,
        node_cmap_range=node_cmap_range,
        node_radius_range=node_radius_range,
        edge_color=edge_color,
        edge_radius=edge_radius,
        edge_cmap=edge_cmap,
        edge_cmap_range=edge_cmap_range,
        edge_radius_range=edge_radius_range,
        projection=projection,
        surf_opacity=surf_opacity,
        hemisphere_slack=hemisphere_slack,
        off_screen=off_screen,
        theme=theme,
    )
    return {**params, **{"plotter": p}}
