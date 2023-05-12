# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Brain surface plotting
~~~~~~~~~~~~~~~~~~~~~~
Brain surface plotting utilities.
"""
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pyvista as pv
from pyvista.plotting.helpers import view_vectors

from .surf import (
    CortexTriSurface,
    ProjectedPolyData,
    make_cmap,
    cortex_theme,
)


def half_width(
    p: pv.Plotter,
    slack: float = 1.05,
) -> Tuple[float, float, float]:
    """
    Return the half-width of the plotter's bounds, multiplied by a slack
    factor.

    We use this to set the position of the camera when we're using
    ``cortex_cameras`` and we receive a string corresponding to an anatomical
    direction (e.g. "dorsal", "ventral", etc.) as the ``position`` argument.

    The slack factor is used to ensure that the camera is not placed exactly
    on the edge of the plotter bounds, which can cause clipping of the
    surface.
    """
    return (
        (p.bounds[1] - p.bounds[0]) / 2 * slack,
        (p.bounds[3] - p.bounds[2]) / 2 * slack,
        (p.bounds[5] - p.bounds[4]) / 2 * slack,
    )


def cortex_view_dict() -> Dict[str, Tuple[Sequence[float], Sequence[float]]]:
    """
    Return a dict containing tuples of (vector, viewup) pairs for each
    hemisphere. The vector is the position of the camera, and the
    viewup is the direction of the "up" vector in the camera frame.
    """
    common = {
        "dorsal": ((0, 0, 1), (1, 0, 0)),
        "ventral": ((0, 0, -1), (-1, 0, 0)),
        "anterior": ((0, 1, 0), (0, 0, 1)),
        "posterior": ((0, -1, 0), (0, 0, 1)),
    }
    return {
        "left": {
            "lateral": ((-1, 0, 0), (0, 0, 1)),
            "medial": ((1, 0, 0), (0, 0, 1)),
            **common,
        },
        "right": {
            "lateral": ((1, 0, 0), (0, 0, 1)),
            "medial": ((-1, 0, 0), (0, 0, 1)),
            **common,
        },
        "both": {
            "left": ((-1, 0, 0), (0, 0, 1)),
            "right": ((1, 0, 0), (0, 0, 1)),
            **common,
        }
    }


def cortex_cameras(
    position: Union[str, Sequence[Tuple[float, float, float]]],
    plotter: pv.Plotter,
    negative: bool = False,
    hemi: Optional[Literal["left", "right"]] = None,
) -> Tuple[
        Tuple[float, float, float],
        Tuple[float, float, float],
        Tuple[float, float, float]
    ]:
    """
    Return a tuple of (position, focal_point, view_up) for the camera. This
    function accepts a string corresponding to an anatomical direction (e.g.
    "dorsal", "ventral", etc.) as the ``position`` argument, and returns the
    corresponding camera position, focal point, and view up vector.
    """
    if isinstance(position, str):
        try:
            #TODO: I'm a little bit concerned that ``view_vectors`` is not
            #      part of the public API. We should probably find a better
            #      way to do this.
            position=view_vectors(
                view=position,
                negative=negative
            )
        except ValueError as e:
            if isinstance(hemi, str):
                try:
                    vector, view_up = cortex_view_dict()[hemi][position]
                    hw = half_width(plotter)
                    focal_point = plotter.center
                    vector = np.array(vector) * hw + focal_point
                    return (vector, focal_point, view_up)
                except KeyError:
                    raise e
            else:
                raise e
    return position


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


def plot_to_display(
    p: pv.Plotter,
    cpos: Optional[Sequence[Sequence[float]]] = "yz",
) -> None:
    p.show(cpos=cpos)


def format_position_as_string(
    position: Union[str, Sequence[Tuple[float, float, float]]],
    precision: int = 2,
    delimiter: str = "x",
) -> str:
    def _fmt_field(field: float) -> str:
        return delimiter.join(
            str(round(v, precision))
            if v >= 0 else f"neg{abs(round(v, precision))}"
            for v in field
        )

    if isinstance(position, str):
        return f"view-{position}"
    elif isinstance(position[0], float) or isinstance(position[0], int):
        return f"vector-{_fmt_field(position)}"
    else:
        return (
            f"vector-{_fmt_field(position[0])}_"
            f"focus-{_fmt_field(position[1])}_"
            f"viewup-{_fmt_field(position[2])}"
        )


def plot_to_image(
    p: pv.Plotter,
    views: Sequence = ("medial", "lateral", "dorsal", "ventral", "anterior", "posterior"),
    window_size: Tuple[int, int] = (1920, 1080),
    basename: Optional[str] = None,
    hemi: Optional[Literal["left", "right", "both"]] = None,
) -> Tuple[np.ndarray]:
    off_screen_orig = p.off_screen
    p.off_screen = True
    if basename is None:
        screenshot = [True] * len(views)
    else:
        screenshot = [
            f"{basename}_{format_position_as_string(cpos)}.png"
            for cpos in views
        ]
    ret = []
    try:
        p.remove_scalar_bar()
    except IndexError:
        pass
    for cpos, fname in zip(views, screenshot):
        p.camera.zoom("tight")
        p.show(
            cpos=cortex_cameras(cpos, plotter=p, hemi=hemi),
            auto_close=False,
        )
        img = p.screenshot(fname, window_size=window_size, return_img=True)
        ret.append(img)
    p.off_screen = off_screen_orig
    p.close()
    return tuple(ret)