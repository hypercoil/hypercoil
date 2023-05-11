# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Brain volume plotting
~~~~~~~~~~~~~~~~~~~~~
Brain volume plotting utilities.
"""
from typing import Any, Optional

import pyvista as pv
import numpy as np
import nibabel as nb

from .surf import CortexTriSurface


def plot_embedded_volume(
    surf: "CortexTriSurface",
    nii: nb.Nifti1Image,
    threshold: float = 0.0,
    projection: Optional[str] = "pial",
    off_screen: bool = False,
    surf_opacity: float = 0.3,
    theme: Optional[Any] = None,
    point_size: Optional[float] = None,
    cmap: Optional[str] = None,
    clim: Optional[tuple] = None,
):
    vol = nii.get_fdata()
    loc = np.where(vol > threshold)
    val = vol[loc]
    coor = np.stack(loc)
    coor = (nii.affine @ np.concatenate(
        (coor, np.ones((1, coor.shape[-1])))
    ))[:3]
    voxdim = nii.header.get_zooms()
    point_size = point_size or voxdim[0] / 2
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
    p.show()

