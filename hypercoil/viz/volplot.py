# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Brain volume plotting
~~~~~~~~~~~~~~~~~~~~~
Brain volume plotting utilities.
"""
from typing import Any, Optional, Sequence

import pyvista as pv
import numpy as np

from .surf import CortexTriSurface
from .utils import cortex_theme


def plot_embedded_volume(
    *,
    surf: "CortexTriSurface",
    coor: np.ndarray,
    val: np.ndarray,
    voxdim: Sequence,
    projection: Optional[str] = "pial",
    off_screen: bool = True,
    surf_opacity: float = 0.3,
    theme: Optional[Any] = None,
    point_size: Optional[float] = None,
    cmap: Optional[str] = None,
    clim: Optional[tuple] = None,
):
    #TODO: cortex_theme doesn't work here for some reason. If the background
    #      is transparent, all of the points are also made transparent. So
    #      we're sticking with a white background for now.
    theme = theme or pv.themes.DocumentTheme() #cortex_theme()
    point_size = point_size or min(voxdim[:3])
    p = pv.Plotter(off_screen=off_screen, theme=theme)

    surf.left.project(projection)
    surf.right.project(projection)
    p.add_mesh(
        surf.left,
        opacity=surf_opacity,
        show_edges=False,
        color='white',
    )
    p.add_mesh(
        surf.right,
        opacity=surf_opacity,
        show_edges=False,
        color='white',
    )
    p.add_points(
        coor.T,
        render_points_as_spheres=False,
        style='points_gaussian',
        emissive=False,
        scalars=val,
        opacity=0.99,
        point_size=point_size,
        ambient=1.0,
        cmap=cmap,
        clim=clim,
    )
    return p
