# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Brain network plotting
~~~~~~~~~~~~~~~~~~~~~~
Brain network plotting utilities.
"""
from typing import Any, Optional, Tuple, Union

import numpy as np
import pyvista as pv
import pandas as pd
from matplotlib.cm import get_cmap

from .surf import CortexTriSurface
from .utils import cortex_theme


def filter_adjacency_data(
    adj: np.ndarray,
    threshold: Union[float, int] = 0.0,
    percent_threshold: bool = False,
    topk_threshold_nodewise: bool = False,
    absolute_threshold: bool = True,
    node_selection: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    adj_incl = np.ones_like(adj, dtype=bool)

    sgn = np.sign(adj)
    if absolute_threshold:
        adj = np.abs(adj)
    if node_selection is not None:
        adj_incl[~node_selection, :] = 0
    if topk_threshold_nodewise:
        indices = np.argpartition(adj, int(threshold), axis=-1)
        indices = indices[..., int(threshold):]
        adj_incl[np.arange(adj.shape[0], dtype=int).reshape(-1, 1), indices] = 0
    elif percent_threshold:
        adj_incl[adj < np.percentile(adj[adj_incl], 100 * threshold)] = 0
    else:
        adj_incl[adj < threshold] = 0

    indices_incl = np.triu_indices(adj.shape[0], k=1)
    adj_incl = adj_incl | adj_incl.T
    adj_incl = adj_incl[indices_incl]
    indices_incl = tuple(i[adj_incl] for i in indices_incl)

    adj = adj[indices_incl]
    sgn = sgn[indices_incl]

    return pd.DataFrame({
        "i": indices_incl[0],
        "j": indices_incl[1],
        "adj": adj,
        "sgn": sgn,
    })


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
    node_cmap_range: Tuple[float, float] = None,
    node_radius_range: Tuple[float, float] = (2, 10),
    edge_color: Optional[str] = "sgn",
    edge_radius: Union[float, str] = "adj",
    edge_cmap: Any = "coolwarm",
    edge_cmap_range: Tuple[float, float] = None,
    edge_radius_range: Tuple[float, float] = (0.1, 1.8),
    projection: Optional[str] = "pial",
    surf_opacity: float = 0.1,
    hemisphere_slack: float = 1.1,
    off_screen: bool = True,
    theme: Optional[pv.themes.DocumentTheme] = None,
) -> pv.Plotter:
    def get_node_color(color):
        # print(color)
        # print(isinstance(color, float))
        # print(isinstance(color, int))
        # print(type(color))
        if isinstance(color, str) or isinstance(color, tuple) or isinstance(color, list):
            return color
        else:
            return get_cmap(node_cmap)(color)

    def get_edge_color(color):
        if isinstance(color, str) or isinstance(color, tuple) or isinstance(color, list):
            return color
        else:
            return get_cmap(edge_cmap)(color)

    def process_edge_values():
        start = coor[edge_values["i"]]
        end = coor[edge_values["j"]]
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
            'parcellation',
            'inflated',
        )

    surf.left.project(projection)
    surf.right.project(projection)

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
        #print(col, get_node_color(col))
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
