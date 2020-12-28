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

    psi_0_2d = functools.partial(functions.psi_gauss_2d, a=1, x_0=0.0, y_0=0.0, k_0=0.0)

    L = 10
    fig = mlab.figure()

    x, y = np.mgrid[-L:L:complex(0, resolution), -L:L:complex(0, resolution)]
    prob_2d = np.abs(psi_0_2d(x, y)) ** 2
    p = mlab.mesh(x, y, prob_2d, representation="wireframe", extent=[-3, 3, -3, 3, 0.0, 1.0])

    ax = mlab.axes(line_width=2, nb_labels=5)
    mlab.title("")
    mlab.show()

