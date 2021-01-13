#!/usr/bin/env python

"""
Functions for Potential and initial wave function psi_0

author: Daniel Scheiermann
email: daniel.scheiermann@stud.uni-hannover.de
license: MIT
Please feel free to use and modify this, but keep the above information. Thanks!
"""

import functools
import numpy as np

from mayavi import mlab
from tvtk.api import tvtk
from tvtk.api import write_data

from supersolids import Animation, functions


# Script runs, if script is run as main script (called by python *.py)
if __name__ == "__main__":
    # due to fft of the points the resolution needs to be 2 ** resolution_exponent
    datapoints_exponent: int = 6
    resolution: int = 2 ** datapoints_exponent

    psi_0_2d = functools.partial(functions.psi_gauss_2d, a=1, x_0=0.0, y_0=0.0, k_0=0.0)
    psi_0_3d = functools.partial(functions.psi_gauss_3d, a=1, x_0=0.0, y_0=0.0, z_0=0.0, k_0=0.0)

    R = 3
    resolution_r = 10
    resolution_phi = 50
    resolution_z = 20
    fig = mlab.figure()

    # 3D example with cylindrical data
    r, theta, z = np.mgrid[0.01:R:complex(0, resolution_r),
                           0:2.0*np.pi:complex(0, resolution_phi),
                           -1:1:complex(0, resolution_z)]
    x, y = r * np.cos(theta), r * np.sin(theta)
    scalar = np.abs(psi_0_3d(x, y, z)) ** 2
    pts = np.empty(z.shape + (3,))
    pts[..., 0], pts[..., 1], pts[..., 2] = x, y, z
    # numpy and vtk have different ordering, so transpose is needed
    pts = pts.transpose(2, 1, 0, 3).copy()
    pts.shape = pts.size//3, 3
    cylinder_grid = tvtk.StructuredGrid(dimensions=x.shape)
    cylinder_grid.points = pts
    cylinder_grid.point_data.scalars = np.ravel(scalar.T.copy())
    cylinder_grid.point_data.scalars.name = "scalars"
    src = mlab.pipeline.add_dataset(cylinder_grid, opacity=0.6)

    plane = mlab.pipeline.grid_plane(src)
    plane.grid_plane.axis = "z"
    plane.grid_plane.position = resolution_z/2

    c_plane = mlab.pipeline.contour_grid_plane(src)
    c_plane.enable_contours = True

    cut = mlab.pipeline.scalar_cut_plane(src)

    iso = mlab.pipeline.iso_surface(src, opacity=0.6)
    write_data(cylinder_grid, "cylinder_grid.vtk")

    ax = mlab.axes(line_width=2, nb_labels=5)
    ax.axes.visibility = False
    mlab.title("")
    mlab.savefig("cylincrical.png")
    mlab.show()

