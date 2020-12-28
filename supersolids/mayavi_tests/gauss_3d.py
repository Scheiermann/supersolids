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
if __name__ == '__main__':
    # due to fft of the points the resolution needs to be 2 ** datapoints_exponent
    datapoints_exponent: int = 6
    resolution: int = 2 ** datapoints_exponent

    psi_0_3d = functools.partial(functions.psi_gauss_3d, a=1, x_0=0.0, y_0=0.0, z_0=0.0, k_0=0.0)

    # testing for 2d plot
    L = 10
    fig = mlab.figure()

    xx, yy, zz = np.mgrid[-L:L:complex(0, resolution), -L:L:complex(0, resolution), -L:L:complex(0, resolution)]
    prob_3d = 1000 * np.abs(psi_0_3d(xx, yy, zz)) ** 2
    t = mlab.contour3d(xx, yy, zz, prob_3d, colormap="spectral", opacity=0.5, transparent=True)
    slice_x = mlab.volume_slice(xx, yy, zz, prob_3d, colormap="spectral",
                                plane_orientation='x_axes', slice_index=resolution/2, extent=[-L, L, -L, L, 0.0, 5.0])
    slice_y = mlab.volume_slice(xx, yy, zz, prob_3d, colormap="spectral",
                                plane_orientation='y_axes', slice_index=resolution/2, extent=[-L, L, -L, L, 0.0, 5.0])

    ax = mlab.axes(line_width=2, nb_labels=5)
    mlab.title("")
    mlab.show()

