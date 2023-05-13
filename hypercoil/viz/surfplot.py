# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Brain surface plotting
~~~~~~~~~~~~~~~~~~~~~~
Brain surface plotting utilities.
"""
from typing import Any, Literal, Optional, Tuple
import pyvista as pv

from .surf import (
    CortexTriSurface,
    ProjectedPolyData,
)
from .utils import cortex_theme


def plot_surf_scalars(
    *,
    surf: "CortexTriSurface",
    projection: str = "veryinflated",
    scalars: str = "parcellation",
    hemi: Optional[Literal["left", "right"]] = None,
    boundary_color: str = "black",
    boundary_width: int = 0,
    off_screen: bool = True,
    cmap: Any = (None, None),
    clim: Any = (None, None),
    theme: Optional[pv.themes.DocumentTheme] = None,
) -> Tuple[Optional[pv.Plotter], Optional[pv.Plotter]]:
    """
    Create plotters for the left and right hemispheres of the surface, with
    specified scalar values plotted on top of the surface.
    """
    def _plot_scalars_hemi(
        hemi_id: Literal["left", "right"],
        hemi_surf: "ProjectedPolyData",
        hemi_cmap: Any,
        hemi_clim: Tuple[float, float],
    ) -> Optional[pv.Plotter]:
        """
        Helper function to plot scalars for a single hemisphere.
        """
        if hemi_id not in hemi:
            p = None
        else:
            p = pv.Plotter(off_screen=off_screen, theme=theme)

            hemi_surf.project(projection)
            p.add_mesh(
                hemi_surf,
                opacity=1.0,
                show_edges=False,
                scalars=scalars,
                cmap=hemi_cmap,
                clim=hemi_clim,
                below_color='black',)
            if boundary_width > 0:
                p.add_mesh(
                    hemi_surf.contour(
                        isosurfaces=range(
                            int(max(hemi_surf.point_data[scalars]))
                        )
                    ),
                    color=boundary_color,
                    line_width=boundary_width,
                )
        return p

    hemi = (hemi,) if hemi is not None else ("left", "right")
    if len(cmap) == 2:
        cmap_left, cmap_right = cmap
    else:
        cmap_left = cmap_right = cmap
    if len(clim) == 2 and isinstance(clim[0], (tuple, list)):
        clim_left, clim_right = clim
    else:
        clim_left = clim_right = clim

    if theme is None:
        theme = cortex_theme()

    pl = _plot_scalars_hemi(
        "left", surf.left, cmap_left, clim_left)
    pr = _plot_scalars_hemi(
        "right", surf.right, cmap_right, clim_right)
    return pl, pr
